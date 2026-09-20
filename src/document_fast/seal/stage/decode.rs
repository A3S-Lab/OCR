use a3s_use_core::UseResult;
use rayon::prelude::*;
use tokio_util::sync::CancellationToken;

use super::runtime_error;
use super::state::{ViewReference, ViewResult};
use crate::cancellation::check_cancelled;
use crate::document_fast::seal::decoder::{
    decode_adjacent_boundary_continuation, decode_page_layout_images, decode_page_views,
};
use crate::document_fast::seal::profile::PicodetLayoutProfile;
use crate::document_fast::seal::refinement::model_ranked_refinement_views;
use crate::document_fast::wired::PixelRect;

struct DecodedView {
    result: ViewResult,
    trace: Option<RefinementTrace>,
}

struct RefinementTrace {
    page_index: usize,
    accepted: usize,
    source_region: PixelRect,
    refinement_regions: Vec<PixelRect>,
}

pub(super) fn decode_inferred_views(
    views: Vec<ViewReference>,
    values: &[f32],
    profile: PicodetLayoutProfile,
    collect_refinement_views: bool,
    cancellation: &CancellationToken,
) -> UseResult<Vec<ViewResult>> {
    decode_inferred_views_with_strategy(
        views,
        values,
        profile,
        collect_refinement_views,
        cancellation,
        parallel_decode_enabled(),
    )
}

fn decode_inferred_views_with_strategy(
    views: Vec<ViewReference>,
    values: &[f32],
    profile: PicodetLayoutProfile,
    collect_refinement_views: bool,
    cancellation: &CancellationToken,
    parallel: bool,
) -> UseResult<Vec<ViewResult>> {
    let sample_elements = profile
        .location_count()
        .checked_mul(profile.output_width())
        .ok_or_else(|| runtime_error("PicoDet output sample cardinality overflowed."))?;
    let expected_elements = views
        .len()
        .checked_mul(sample_elements)
        .ok_or_else(|| runtime_error("PicoDet output batch cardinality overflowed."))?;
    if values.len() != expected_elements {
        return Err(runtime_error(format!(
            "PicoDet output batch requires {expected_elements} values for {} views, found {}.",
            views.len(),
            values.len()
        )));
    }

    let trace_refinement_plan = std::env::var_os("A3S_OCR_TRACE_SEAL_DETECTIONS").is_some();
    let decode = |(sample, reference): (usize, ViewReference)| {
        check_cancelled(cancellation)?;
        let start = sample * sample_elements;
        let raw_output = &values[start..start + sample_elements];
        decode_view(
            reference,
            raw_output,
            profile,
            collect_refinement_views,
            trace_refinement_plan,
        )
    };
    let decoded = if parallel {
        views
            .into_par_iter()
            .enumerate()
            .map(decode)
            .collect::<Vec<_>>()
    } else {
        views.into_iter().enumerate().map(decode).collect()
    };

    // Indexed Rayon collection preserves input order. Resolve failures and
    // diagnostics in that same order so parallel scheduling cannot change the
    // batch's observable result or the first reported error.
    decoded
        .into_iter()
        .map(|decoded| {
            let decoded = decoded?;
            if let Some(trace) = decoded.trace {
                eprintln!(
                    "A3S_OCR_SEAL_REFINEMENT_PLAN page_index={} accepted={} source_region={:?} refinement_regions={:?}",
                    trace.page_index,
                    trace.accepted,
                    trace.source_region,
                    trace.refinement_regions,
                );
            }
            Ok(decoded.result)
        })
        .collect()
}

fn decode_view(
    reference: ViewReference,
    raw_output: &[f32],
    profile: PicodetLayoutProfile,
    collect_refinement_views: bool,
    trace_refinement_plan: bool,
) -> UseResult<DecodedView> {
    let seals = decode_page_views(&[(&reference.view, raw_output)], &reference.image, profile)
        .and_then(|mut seals| {
            if let Some(predecessor) = reference.adjacent_boundary {
                if let Some(continuation) = decode_adjacent_boundary_continuation(
                    raw_output,
                    reference.view,
                    &reference.image,
                    profile,
                    predecessor,
                )? {
                    seals.push(continuation);
                }
            }
            Ok(seals)
        });
    let layout_images = if collect_refinement_views {
        decode_page_layout_images(raw_output, reference.view, &reference.image, profile)
    } else {
        Ok(Vec::new())
    };
    let (refinement_views, trace) = if collect_refinement_views {
        match &seals {
            Ok(accepted) => {
                let planned = model_ranked_refinement_views(
                    raw_output,
                    reference.view,
                    &reference.image,
                    profile,
                    accepted,
                )?;
                let trace = trace_refinement_plan.then(|| RefinementTrace {
                    page_index: reference.page_index,
                    accepted: accepted.len(),
                    source_region: reference.view.region,
                    refinement_regions: planned.iter().map(|view| view.region).collect(),
                });
                let views = planned
                    .into_iter()
                    .map(|view| ViewReference {
                        page_index: reference.page_index,
                        image: reference.image.clone(),
                        view,
                        adjacent_boundary: None,
                    })
                    .collect();
                (views, trace)
            }
            Err(_) => (Vec::new(), None),
        }
    } else {
        (Vec::new(), None)
    };
    Ok(DecodedView {
        result: ViewResult {
            page_index: reference.page_index,
            seals,
            layout_images,
            refinement_views,
        },
        trace,
    })
}

fn parallel_decode_enabled() -> bool {
    // The detailed decoder diagnostics are emitted inside the decoder itself.
    // Keep that opt-in diagnostic mode serial so its records remain ordered.
    if std::env::var_os("A3S_OCR_TRACE_SEAL_DETECTIONS").is_some()
        || std::env::var_os("A3S_OCR_TRACE_SEAL_REFINEMENT").is_some()
    {
        return false;
    }
    #[cfg(test)]
    if std::env::var_os("A3S_OCR_TEST_DISABLE_PARALLEL_SEAL_DECODE").is_some() {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use image::RgbImage;

    use super::*;
    use crate::document_fast::seal::preprocess::full_page_view;

    #[test]
    fn parallel_decode_preserves_serial_results_and_input_order() {
        let profile = PicodetLayoutProfile::Small;
        let image = Arc::new(RgbImage::new(480, 480));
        let page_indices = [8, 3, 11, 1, 6, 0, 9, 4];
        let views = page_indices
            .into_iter()
            .map(|page_index| ViewReference {
                page_index,
                image: image.clone(),
                view: full_page_view(&image),
                adjacent_boundary: None,
            })
            .collect::<Vec<_>>();
        let sample_elements = profile.location_count() * profile.output_width();
        let mut values = vec![0.0; views.len() * sample_elements];
        for sample_index in 0..views.len() {
            let offset = sample_index as f32;
            let sample =
                &mut values[sample_index * sample_elements..(sample_index + 1) * sample_elements];
            sample[..7].copy_from_slice(&[
                20.0 + offset,
                30.0,
                100.0 + offset,
                120.0,
                0.1,
                0.2,
                0.9,
            ]);
            sample[7..14].copy_from_slice(&[220.0, 230.0, 300.0, 320.0, 0.8, 0.1, 0.2]);
        }
        let cancellation = CancellationToken::new();
        let serial = decode_inferred_views_with_strategy(
            views.clone(),
            &values,
            profile,
            true,
            &cancellation,
            false,
        )
        .unwrap();
        let parallel =
            decode_inferred_views_with_strategy(views, &values, profile, true, &cancellation, true)
                .unwrap();

        assert_eq!(
            parallel
                .iter()
                .map(|result| result.page_index)
                .collect::<Vec<_>>(),
            page_indices
        );
        for (serial, parallel) in serial.into_iter().zip(parallel) {
            assert_eq!(serial.page_index, parallel.page_index);
            assert_eq!(serial.seals.unwrap(), parallel.seals.unwrap());
            assert_eq!(
                serial.layout_images.unwrap(),
                parallel.layout_images.unwrap()
            );
            let view_identity =
                |view: ViewReference| (view.page_index, view.view, view.adjacent_boundary);
            assert_eq!(
                serial
                    .refinement_views
                    .into_iter()
                    .map(view_identity)
                    .collect::<Vec<_>>(),
                parallel
                    .refinement_views
                    .into_iter()
                    .map(view_identity)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn decode_rejects_a_truncated_batch_without_indexing_past_it() {
        let profile = PicodetLayoutProfile::Small;
        let image = Arc::new(RgbImage::new(64, 64));
        let views = vec![ViewReference {
            page_index: 0,
            image: image.clone(),
            view: full_page_view(&image),
            adjacent_boundary: None,
        }];
        let values = vec![0.0; profile.location_count() * profile.output_width() - 1];
        let error = match decode_inferred_views_with_strategy(
            views,
            &values,
            profile,
            false,
            &CancellationToken::new(),
            true,
        ) {
            Ok(_) => panic!("a truncated output batch must be rejected"),
            Err(error) => error,
        };

        assert_eq!(error.code, "use.ocr.runtime_failed");
        assert!(error.message.contains("requires"));
    }
}
