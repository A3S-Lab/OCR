use a3s_use_core::{Readiness, UseError, UseResult};
use async_trait::async_trait;

use super::assets::SlanetPlusAssets;
use super::projection::table_evidence;
use super::stage::{DetectedPage, TableStageBatch, TableStageRunner};
use crate::cancellation::CancellationScope;
use crate::{
    OcrInput, OcrProvider, OcrProviderBatchOutput, OcrProviderBatchRequest,
    OcrProviderBatchSlotOutput, OcrProviderDescriptor, OcrProviderOutput, OcrProviderStatus,
    OcrStage, OcrStageOutcome, PpOcrV6Provider,
};

pub const DOCUMENT_FAST_PROVIDER_ID: &str = "document-fast-v1";
const ENGINE_NAME: &str = "a3s-power-native";
const DOCUMENT_FAST_MODEL: &str = "pp-ocr-v6-small+slanet-plus-wired";

/// Local fast-document composition with PP-OCRv6 text and model-backed wired
/// table structure. Cross-page reconciliation remains a Parser concern.
#[derive(Clone)]
pub struct DocumentFastOcrProvider {
    descriptor: OcrProviderDescriptor,
    text: PpOcrV6Provider,
    table: TableStageRunner,
}

impl DocumentFastOcrProvider {
    pub fn from_env() -> UseResult<Self> {
        let table_assets = SlanetPlusAssets::from_env()?;
        Ok(Self {
            descriptor: OcrProviderDescriptor::new(DOCUMENT_FAST_PROVIDER_ID, ENGINE_NAME, false)?
                .with_stages(vec![
                    OcrStage::Preprocessing,
                    OcrStage::Text,
                    OcrStage::Table,
                ])?,
            text: PpOcrV6Provider::from_env()?,
            table: TableStageRunner::new(table_assets)?,
        })
    }
}

#[async_trait]
impl OcrProvider for DocumentFastOcrProvider {
    fn descriptor(&self) -> OcrProviderDescriptor {
        self.descriptor.clone()
    }

    fn diagnostic(&self) -> OcrProviderStatus {
        let text = self.text.diagnostic();
        if text.readiness == Readiness::Ready {
            OcrProviderStatus {
                readiness: Readiness::Ready,
                model: Some(DOCUMENT_FAST_MODEL.to_string()),
                model_dir: Some(self.table.model_root().to_path_buf()),
                message: "Local PP-OCRv6 text and SLANet-Plus wired-table models are ready."
                    .to_string(),
                suggestions: Vec::new(),
            }
        } else {
            OcrProviderStatus {
                readiness: text.readiness,
                model: Some(DOCUMENT_FAST_MODEL.to_string()),
                model_dir: text.model_dir,
                message: format!(
                    "The wired-table model is ready, but the PP-OCRv6 text model is not: {}",
                    text.message
                ),
                suggestions: text.suggestions,
            }
        }
    }

    async fn recognize(&self, input: OcrInput) -> UseResult<OcrProviderOutput> {
        self.text.recognize(input).await
    }

    async fn recognize_batch(
        &self,
        request: OcrProviderBatchRequest,
    ) -> UseResult<OcrProviderBatchOutput> {
        let stages = request.stages;
        let slot_ids = request
            .slots
            .iter()
            .map(|slot| slot.slot_id.clone())
            .collect::<Vec<_>>();
        let text_stages = stages
            .iter()
            .copied()
            .filter(|stage| matches!(stage, OcrStage::Preprocessing | OcrStage::Text))
            .collect::<Vec<_>>();
        let text_request = (!text_stages.is_empty()).then(|| OcrProviderBatchRequest {
            stages: text_stages,
            slots: request.slots.clone(),
        });
        let table_slots = stages.contains(&OcrStage::Table).then_some(request.slots);
        let cancellation = CancellationScope::new();
        let token = cancellation.token();
        let text_future = async {
            match text_request {
                Some(request) => Some(self.text.recognize_batch(request).await),
                None => None,
            }
        };
        let table_future = async {
            match table_slots {
                Some(slots) => Some(self.table.run(slots, token).await),
                None => None,
            }
        };
        let (text_result, table_result) = tokio::join!(text_future, table_future);
        cancellation.disarm();

        let (text_slots, mut execution_receipts) = normalize_text_slots(text_result, &slot_ids)?;
        let (table_slots, table_receipts) = normalize_table_slots(table_result, &slot_ids)?;
        execution_receipts.extend(table_receipts);
        let mut outputs = Vec::with_capacity(slot_ids.len());
        for (index, slot_id) in slot_ids.into_iter().enumerate() {
            let text_slot = text_slots.get(index).and_then(Option::as_ref);
            let mut output = text_slot.and_then(|slot| slot.output.clone());
            let table_resolution = match table_slots
                .get(index)
                .and_then(Option::as_ref)
                .map(|slot| slot.as_ref())
            {
                None => None,
                Some(Err(error)) => Some(Err(error.clone())),
                Some(Ok(page)) => {
                    let page = clone_page(page);
                    Some(
                        table_evidence(page, output.as_ref()).map(|(evidence, receipts)| {
                            let output = output.get_or_insert_with(OcrProviderOutput::default);
                            output.model = Some(DOCUMENT_FAST_MODEL.to_string());
                            output.execution_receipts.extend(receipts);
                            evidence
                        }),
                    )
                }
            };
            let outcomes = stages
                .iter()
                .map(|stage| match stage {
                    OcrStage::Preprocessing | OcrStage::Text => text_slot
                        .and_then(|slot| slot.stages.iter().find(|outcome| outcome.stage == *stage))
                        .cloned()
                        .unwrap_or_else(|| {
                            OcrStageOutcome::failed(
                                *stage,
                                composition_error(
                                    "The PP-OCRv6 sub-provider omitted a requested stage outcome.",
                                ),
                            )
                        }),
                    OcrStage::Table => match &table_resolution {
                        Some(Ok(evidence)) => {
                            OcrStageOutcome::completed_with_evidence(evidence.clone())
                        }
                        Some(Err(error)) => OcrStageOutcome::failed(*stage, error.clone()),
                        None => OcrStageOutcome::failed(
                            *stage,
                            composition_error(
                                "The table sub-provider omitted a requested stage outcome.",
                            ),
                        ),
                    },
                    OcrStage::Orientation
                    | OcrStage::Layout
                    | OcrStage::Formula
                    | OcrStage::Seal => OcrStageOutcome::unsupported(*stage),
                })
                .collect();
            outputs.push(OcrProviderBatchSlotOutput {
                slot_id,
                stages: outcomes,
                output,
            });
        }
        Ok(OcrProviderBatchOutput {
            slots: outputs,
            execution_receipts,
        })
    }
}

fn normalize_text_slots(
    result: Option<UseResult<OcrProviderBatchOutput>>,
    expected: &[crate::OcrBatchSlotId],
) -> UseResult<(
    Vec<Option<OcrProviderBatchSlotOutput>>,
    Vec<crate::OcrExecutionReceipt>,
)> {
    let Some(result) = result else {
        return Ok(((0..expected.len()).map(|_| None).collect(), Vec::new()));
    };
    let output = result?;
    if output.slots.len() != expected.len()
        || output
            .slots
            .iter()
            .zip(expected)
            .any(|(slot, expected)| slot.slot_id != *expected)
    {
        return Err(composition_error(
            "The PP-OCRv6 sub-provider changed document-fast slot identity or cardinality.",
        ));
    }
    Ok((
        output.slots.into_iter().map(Some).collect(),
        output.execution_receipts,
    ))
}

type TableSlot = Result<DetectedPage, UseError>;

fn normalize_table_slots(
    result: Option<UseResult<TableStageBatch>>,
    expected: &[crate::OcrBatchSlotId],
) -> UseResult<(Vec<Option<TableSlot>>, Vec<crate::OcrExecutionReceipt>)> {
    let Some(result) = result else {
        return Ok(((0..expected.len()).map(|_| None).collect(), Vec::new()));
    };
    match result {
        Ok(output) => {
            if output.slots.len() != expected.len()
                || output
                    .slots
                    .iter()
                    .zip(expected)
                    .any(|(slot, expected)| slot.slot_id != *expected)
            {
                return Err(composition_error(
                    "The table sub-provider changed document-fast slot identity or cardinality.",
                ));
            }
            Ok((
                output
                    .slots
                    .into_iter()
                    .map(|slot| Some(slot.page))
                    .collect(),
                output.receipts,
            ))
        }
        Err(error) => Ok((
            expected.iter().map(|_| Some(Err(error.clone()))).collect(),
            Vec::new(),
        )),
    }
}

fn clone_page(page: &DetectedPage) -> DetectedPage {
    DetectedPage {
        canvas: page.canvas,
        tables: page
            .tables
            .iter()
            .map(|table| super::stage::DetectedTable {
                region: table.region,
                grid: table.grid.clone(),
            })
            .collect(),
        receipts: page.receipts.clone(),
    }
}

fn composition_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.provider_batch_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_provider_declares_only_implemented_stages() {
        let Some(_) = std::env::var_os("A3S_OCR_SLANET_PLUS_MODEL_DIR") else {
            return;
        };
        let provider = DocumentFastOcrProvider::from_env().unwrap();
        assert_eq!(provider.descriptor().id, DOCUMENT_FAST_PROVIDER_ID);
        assert_eq!(
            provider.descriptor().supported_stages,
            vec![OcrStage::Preprocessing, OcrStage::Text, OcrStage::Table]
        );
        assert!(!provider.descriptor().supports_stage(OcrStage::Seal));
        assert_eq!(super::super::assets::MODEL_FAMILY, "slanet-plus-wired");
    }

    #[tokio::test]
    #[ignore = "requires pinned PP-OCRv6/SLANet-Plus bundles and the real table fixture"]
    async fn real_provider_emits_model_backed_grid_geometry_and_cell_text() {
        let fixture_root = std::env::var_os("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR")
            .expect("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR must name the reviewed fixture root");
        let source = std::path::Path::new(&fixture_root).join("page-0002.png");
        let client =
            crate::OcrClient::with_provider(DocumentFastOcrProvider::from_env().unwrap()).unwrap();
        let result = client
            .extract_batch(
                crate::OcrBatchRequest::new(
                    vec![OcrStage::Preprocessing, OcrStage::Text, OcrStage::Table],
                    vec![crate::OcrBatchSlotRequest::new(
                        crate::OcrBatchSlotId::new("page-2").unwrap(),
                        source,
                    )],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(result.slots[0].status, crate::OcrBatchSlotStatus::Completed);
        let table_stage = result.slots[0]
            .stages
            .iter()
            .find(|outcome| outcome.stage == OcrStage::Table)
            .unwrap();
        let crate::OcrStageEvidence::Table(evidence) = table_stage.evidence.as_ref().unwrap()
        else {
            panic!("table stage returned non-table evidence");
        };
        assert_eq!(evidence.tables.len(), 1);
        let table = &evidence.tables[0];
        assert_eq!((table.row_count, table.column_count), (Some(6), Some(6)));
        assert_eq!(table.cells.len(), 29);
        assert!(table.cells.iter().all(|cell| cell.region.is_some()));
        assert!(
            table
                .cells
                .iter()
                .filter(|cell| cell.text.is_some())
                .count()
                >= 20
        );
        let output = result.slots[0].result.as_ref().unwrap();
        assert!(!output.text.is_empty());
        assert!(output
            .execution_receipts
            .iter()
            .any(|receipt| { receipt.model.family == "slanet-plus-wired-encoder" }));
        assert!(result
            .execution_receipts
            .iter()
            .any(|receipt| { receipt.model.family == "slanet-plus-wired-encoder" }));
    }

    #[tokio::test]
    #[ignore = "requires the pinned SLANet-Plus bundle and the real cross-page table fixture"]
    async fn real_provider_batches_cross_page_table_fragments() {
        let fixture_root = std::env::var_os("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR")
            .expect("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR must name the reviewed fixture root");
        let root = std::path::Path::new(&fixture_root);
        let slots = (2..=4)
            .map(|page| {
                crate::OcrBatchSlotRequest::new(
                    crate::OcrBatchSlotId::new(format!("page-{page}")).unwrap(),
                    root.join(format!("page-{page:04}.png")),
                )
            })
            .collect();
        let client =
            crate::OcrClient::with_provider(DocumentFastOcrProvider::from_env().unwrap()).unwrap();
        let result = client
            .extract_batch(crate::OcrBatchRequest::new(vec![OcrStage::Table], slots).unwrap())
            .await
            .unwrap();
        assert_eq!(result.slots.len(), 3);
        let expected = [(6, 6, 29), (8, 7, 25), (3, 6, 17)];
        for (slot, (rows, columns, cells)) in result.slots.iter().zip(expected) {
            assert_eq!(slot.status, crate::OcrBatchSlotStatus::Completed);
            let table_stage = &slot.stages[0];
            let crate::OcrStageEvidence::Table(evidence) = table_stage.evidence.as_ref().unwrap()
            else {
                panic!("table stage returned non-table evidence");
            };
            assert_eq!(evidence.tables.len(), 1);
            let table = &evidence.tables[0];
            assert_eq!(
                (table.row_count, table.column_count),
                (Some(rows), Some(columns))
            );
            assert_eq!(table.cells.len(), cells);
            assert!(table.cells.iter().all(|cell| cell.region.is_some()));
        }
        let table_receipts = result
            .execution_receipts
            .iter()
            .filter(|receipt| receipt.model.family == "slanet-plus-wired-encoder")
            .collect::<Vec<_>>();
        assert_eq!(table_receipts.len(), 1);
        if let Some(expected_device) = std::env::var_os("A3S_OCR_EXPECT_TABLE_DEVICE") {
            assert_eq!(
                table_receipts[0].runtime.device,
                expected_device.to_string_lossy()
            );
        }
    }
}
