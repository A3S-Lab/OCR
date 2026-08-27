//! Model-backed page orientation with a transform-consistency abstention.

mod batching;
mod transform;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use a3s_power::inference::graph::{GraphExecutor, GraphIdentity, GraphPlan};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, ExecutionDigest, ExecutionPermit, InferenceLimits,
    ModelIdentity, TensorInput, WeightStore,
};
use a3s_use_core::{UseError, UseResult};
use image::RgbImage;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use super::page_orientation_preprocess as preprocess;
use super::shared_decode::SharedDecodedImage;
use crate::cancellation::{check_cancelled, run_blocking_with};
use crate::receipt::project_receipt;
use crate::{OcrExecutionReceipt, OcrStage, OcrStageOutcome};

pub(super) use transform::SourceCanvasTransform;

const MODEL_ENV: &str = "A3S_OCR_PAGE_ORIENTATION_MODEL_DIR";
const FAMILY: &str = "pp-lcnet-x1-doc-orientation";
const REVISION: &str = "paddlex-official";
const ROLE: &str = "orientation-classification";
const SOURCE_SHA256: &str = "96e898f047a0e460ba0652e9afb8c874e53872821cfd7a3fec53a5ab62df92f0";
const GRAPH_SHA256: &str = "58af7aa1ccdba05938e409f2b4ce299465740bd615ed76154167527b66389110";
const WEIGHTS_SHA256: &str = "50c0b9a20725346542c87a7b95197b725e7aa379876849d547e57d85006dd3a6";
const WEIGHTS_COLLECTION_SHA256: &str =
    "15fd6132e3afdae4a60457e12847944da3867a3a6c54a5ca89d11a4b2c30017b";
const WEIGHTS_BYTES: u64 = 6_764_348;

#[derive(Debug, Clone)]
pub(super) struct PageOrientationAssets {
    root: PathBuf,
    graph: String,
}

impl PageOrientationAssets {
    pub(super) fn from_env_optional() -> UseResult<Option<Self>> {
        let Some(root) = std::env::var_os(MODEL_ENV).filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        Self::from_root(Path::new(&root)).map(Some)
    }

    fn from_root(root: &Path) -> UseResult<Self> {
        let root = std::fs::canonicalize(root).map_err(|error| {
            model_error(format!(
                "Failed to resolve the page-orientation model directory '{}': {error}",
                root.display()
            ))
        })?;
        if !root.is_dir() {
            return Err(model_error(
                "The page-orientation model root must be a directory.",
            ));
        }
        let graph_path = checked_asset(&root, "graph.json", None, GRAPH_SHA256)?;
        checked_asset(
            &root,
            "model.safetensors",
            Some(WEIGHTS_BYTES),
            WEIGHTS_SHA256,
        )?;
        let graph = std::fs::read_to_string(graph_path).map_err(|error| {
            model_error(format!(
                "Failed to read the page-orientation graph: {error}"
            ))
        })?;
        Ok(Self { root, graph })
    }
}

#[derive(Clone)]
pub(super) struct PageOrientationRunner {
    engine: Arc<Mutex<PageOrientationEngine>>,
}

impl PageOrientationRunner {
    pub(super) fn new(assets: PageOrientationAssets) -> UseResult<Self> {
        Ok(Self {
            engine: Arc::new(Mutex::new(PageOrientationEngine::load(&assets)?)),
        })
    }

    pub(super) async fn normalize_decoded(
        &self,
        images: Vec<SharedDecodedImage>,
        cancellation: CancellationToken,
    ) -> UseResult<PageOrientationBatch> {
        let engine = Arc::clone(&self.engine);
        run_blocking_with(
            "document page-orientation normalization",
            cancellation.clone(),
            move |cancellation| {
                check_cancelled(&cancellation)?;
                let engine = engine.lock().map_err(|_| {
                    orientation_error("The page-orientation engine lock is unavailable.")
                })?;
                engine.normalize(images, &cancellation)
            },
        )
        .await
    }
}

pub(super) struct PageOrientationBatch {
    pub(super) images: Vec<SharedDecodedImage>,
    pub(super) slots: Vec<PageOrientationSlot>,
    pub(super) receipts: Vec<OcrExecutionReceipt>,
}

pub(super) struct PageOrientationSlot {
    pub(super) outcome: OcrStageOutcome,
    pub(super) transform: Option<SourceCanvasTransform>,
    pub(super) receipts: Vec<OcrExecutionReceipt>,
}

#[derive(Clone, Copy)]
struct Classification {
    class: usize,
}

struct PageOrientationEngine {
    runtime: EmbeddedRuntime,
    graph: GraphExecutor,
    identity: ModelIdentity,
}

impl PageOrientationEngine {
    fn load(assets: &PageOrientationAssets) -> UseResult<Self> {
        let limits = InferenceLimits {
            max_concurrent_requests: 1,
            max_queued_requests: 32,
            ..InferenceLimits::default()
        };
        let runtime = EmbeddedRuntime::new(DevicePreference::Auto, limits.clone())
            .map_err(|error| power_error("initialize the orientation runtime", error))?;
        let weights = Arc::new(
            WeightStore::open(&assets.root, &limits)
                .map_err(|error| power_error("open the orientation weights", error))?,
        );
        weights
            .verify_integrity(FAMILY, WEIGHTS_COLLECTION_SHA256)
            .map_err(|error| power_error("verify the orientation weights", error))?;
        let graph_identity = GraphIdentity::new(FAMILY, ROLE, "onnx", SOURCE_SHA256, 17);
        let plan = GraphPlan::parse(&assets.graph, &graph_identity, &weights, &limits)
            .map_err(|error| power_error("validate the orientation graph", error))?;
        let graph = GraphExecutor::new(plan, weights, runtime.clone())
            .map_err(|error| power_error("materialize the orientation graph", error))?;
        Ok(Self {
            runtime,
            graph,
            identity: ModelIdentity::new(FAMILY, REVISION, WEIGHTS_SHA256),
        })
    }

    fn normalize(
        &self,
        images: Vec<SharedDecodedImage>,
        cancellation: &CancellationToken,
    ) -> UseResult<PageOrientationBatch> {
        let permit = self
            .runtime
            .begin(cancellation)
            .map_err(|error| power_error("admit the orientation request", error))?;
        let mut classifications = vec![[None; 4]; images.len()];
        let mut slot_receipts = vec![Vec::new(); images.len()];
        let mut receipts = Vec::new();
        {
            let mut classification_state = ClassificationState {
                classifications: &mut classifications,
                slot_receipts: &mut slot_receipts,
                receipts: &mut receipts,
                permit: &permit,
                cancellation,
            };

            let originals = images
                .iter()
                .enumerate()
                .filter_map(|(slot, image)| {
                    image.as_ref().ok().map(|image| (slot, Arc::clone(image)))
                })
                .collect::<Vec<_>>();
            self.classify_entries(
                originals.into_iter().map(|(slot, image)| (slot, 0, image)),
                &mut classification_state,
            )?;

            let candidates = classification_state
                .classifications
                .iter()
                .enumerate()
                .filter_map(|(slot, classes)| {
                    classes[0]
                        .filter(|classification| classification.class != 0)
                        .map(|_| slot)
                })
                .collect::<Vec<_>>();
            let mut rotations = Vec::with_capacity(candidates.len().saturating_mul(3));
            for slot in candidates {
                let source = images
                    .get(slot)
                    .and_then(|image| image.as_ref().ok())
                    .ok_or_else(|| {
                        orientation_error(
                            "A classified orientation slot no longer retained its decoded image.",
                        )
                    })?;
                rotations.extend([
                    (slot, 1, Arc::clone(source)),
                    (slot, 2, Arc::clone(source)),
                    (slot, 3, Arc::clone(source)),
                ]);
            }
            self.classify_entries(rotations.into_iter(), &mut classification_state)?;
        }

        let mut normalized = Vec::with_capacity(images.len());
        let mut slots = Vec::with_capacity(images.len());
        for (slot, image) in images.into_iter().enumerate() {
            match image {
                Err(error) => {
                    normalized.push(Err(error.clone()));
                    slots.push(PageOrientationSlot {
                        outcome: OcrStageOutcome::failed(OcrStage::Orientation, error),
                        transform: None,
                        receipts: slot_receipts[slot].clone(),
                    });
                }
                Ok(source) => {
                    let classes = classifications[slot];
                    let base = classes[0].ok_or_else(|| {
                        orientation_error("Page-orientation output omitted an admitted slot.")
                    })?;
                    let consistent = base.class == 0
                        || classes.iter().enumerate().all(|(turns, classification)| {
                            classification.is_some_and(|classification| {
                                classification.class == (base.class + turns) % 4
                            })
                        });
                    let quarter_turns = if consistent {
                        ((4 - base.class) % 4) as u8
                    } else {
                        0
                    };
                    let transform =
                        SourceCanvasTransform::new(source.width(), source.height(), quarter_turns)?;
                    let image = match quarter_turns {
                        0 => source,
                        1 => Arc::new(image::imageops::rotate90(source.as_ref())),
                        2 => Arc::new(image::imageops::rotate180(source.as_ref())),
                        3 => Arc::new(image::imageops::rotate270(source.as_ref())),
                        _ => {
                            return Err(orientation_error(
                                "An admitted orientation transform exceeded three quarter turns.",
                            ));
                        }
                    };
                    normalized.push(Ok(image));
                    slots.push(PageOrientationSlot {
                        outcome: if consistent {
                            OcrStageOutcome::completed(OcrStage::Orientation)
                        } else {
                            OcrStageOutcome::skipped(OcrStage::Orientation, orientation_abstained())
                        },
                        transform: Some(transform),
                        receipts: slot_receipts[slot].clone(),
                    });
                }
            }
        }
        Ok(PageOrientationBatch {
            images: normalized,
            slots,
            receipts,
        })
    }

    fn classify_entries(
        &self,
        entries: impl Iterator<Item = (usize, u8, Arc<RgbImage>)>,
        state: &mut ClassificationState<'_>,
    ) -> UseResult<()> {
        let requested_cap = test_orientation_batch_cap().unwrap_or(crate::batch::MAX_BATCH_SLOTS);
        let maximum_batch_size =
            batching::maximum_batch_size(self.runtime.limits(), requested_cap)?;
        let mut pending = Vec::with_capacity(maximum_batch_size);
        for entry in entries {
            pending.push(entry);
            if pending.len() == maximum_batch_size {
                self.classify_batch(&pending, maximum_batch_size, state)?;
                pending.clear();
            }
        }
        if !pending.is_empty() {
            self.classify_batch(&pending, maximum_batch_size, state)?;
        }
        Ok(())
    }

    fn classify_batch(
        &self,
        entries: &[(usize, u8, Arc<RgbImage>)],
        maximum_batch_size: usize,
        state: &mut ClassificationState<'_>,
    ) -> UseResult<()> {
        check_cancelled(state.cancellation)?;
        let images = entries
            .iter()
            .map(|(_, turns, image)| (&**image, *turns))
            .collect::<Vec<_>>();
        let values = preprocess::batch_oriented(&images, maximum_batch_size)?;
        let shape = vec![
            entries.len(),
            3,
            preprocess::INPUT_SIDE,
            preprocess::INPUT_SIDE,
        ];
        let input = TensorInput::new(shape, values, self.runtime.limits())
            .map_err(|error| power_error("validate an orientation input", error))?;
        let input_digest = ExecutionDigest::f32_tensor(&input.shape, &input.values);
        let output = self
            .graph
            .run(input, state.permit, state.cancellation)
            .map_err(|error| power_error("execute the orientation graph", error))?;
        if output.shape != [entries.len(), 4]
            || output.values.len() != entries.len() * 4
            || output.values.iter().any(|value| !value.is_finite())
        {
            return Err(orientation_error(format!(
                "Page-orientation output must be finite [N,4], found {:?}.",
                output.shape
            )));
        }
        let output_digest = ExecutionDigest::f32_tensor(&output.shape, &output.values);
        let receipt = project_receipt(self.runtime.receipt(
            self.identity.clone(),
            input_digest,
            output_digest,
        ));
        for ((slot, turns, _), scores) in entries.iter().zip(output.values.chunks_exact(4)) {
            let class = top1(scores);
            state.classifications[*slot][usize::from(*turns)] = Some(Classification { class });
        }
        let mut covered_slots = entries.iter().map(|(slot, _, _)| *slot).collect::<Vec<_>>();
        covered_slots.sort_unstable();
        covered_slots.dedup();
        for slot in covered_slots {
            state.slot_receipts[slot].push(receipt.clone());
        }
        state.receipts.push(receipt);
        Ok(())
    }
}

struct ClassificationState<'a> {
    classifications: &'a mut [[Option<Classification>; 4]],
    slot_receipts: &'a mut [Vec<OcrExecutionReceipt>],
    receipts: &'a mut Vec<OcrExecutionReceipt>,
    permit: &'a ExecutionPermit,
    cancellation: &'a CancellationToken,
}

fn test_orientation_batch_cap() -> Option<usize> {
    #[cfg(test)]
    if let Some(value) = std::env::var_os("A3S_OCR_TEST_ORIENTATION_MAX_BATCH_SIZE") {
        return value.to_string_lossy().parse::<usize>().ok();
    }
    None
}

fn top1(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}

fn checked_asset(
    root: &Path,
    relative: &str,
    expected_bytes: Option<u64>,
    expected_sha256: &str,
) -> UseResult<PathBuf> {
    let requested = root.join(relative);
    let path = std::fs::canonicalize(&requested).map_err(|error| {
        model_error(format!(
            "Required page-orientation asset '{}' is unreadable: {error}",
            requested.display()
        ))
    })?;
    let metadata = std::fs::metadata(&path).map_err(|error| {
        model_error(format!(
            "Failed to inspect page-orientation asset '{}': {error}",
            path.display()
        ))
    })?;
    if !path.starts_with(root)
        || !metadata.is_file()
        || expected_bytes.is_some_and(|bytes| metadata.len() != bytes)
        || file_sha256(&path)? != expected_sha256
    {
        return Err(model_error(format!(
            "Page-orientation asset '{}' failed its pinned identity.",
            path.display()
        )));
    }
    Ok(path)
}

fn file_sha256(path: &Path) -> UseResult<String> {
    let mut file = std::fs::File::open(path)
        .map_err(|error| model_error(format!("Failed to open '{}': {error}", path.display())))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            model_error(format!("Failed to hash '{}': {error}", path.display()))
        })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn orientation_abstained() -> UseError {
    UseError::new(
        "use.ocr.orientation_abstained",
        "The page-orientation classes were not equivariant under all quarter-turn transforms; the source canvas was left unchanged.",
    )
}

fn model_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.orientation_model_invalid", message)
        .with_suggestion("Restore the exact reviewed local page-orientation model bundle.")
}

fn orientation_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.orientation_failed", message)
}

fn power_error(action: &str, error: impl std::fmt::Display) -> UseError {
    UseError::new(
        "use.ocr.runtime_failed",
        format!("Failed to {action} through a3s-power: {error}"),
    )
}
