use std::sync::Arc;

use a3s_use_core::{UseError, UseResult};
use image::RgbImage;
use rayon::prelude::*;
use tokio_util::sync::CancellationToken;

use crate::cancellation::{check_cancelled, run_blocking_with};
use crate::preprocess::decode_image;
use crate::OcrProviderBatchSlot;

pub(super) type SharedDecodedImage = Result<Arc<RgbImage>, UseError>;

pub(super) async fn decode_slots_once(
    slots: &[OcrProviderBatchSlot],
    cancellation: CancellationToken,
) -> UseResult<Vec<SharedDecodedImage>> {
    let inputs = slots
        .iter()
        .map(|slot| slot.input.clone())
        .collect::<Vec<_>>();
    run_blocking_with(
        "document-fast shared image decoding",
        cancellation,
        move |cancellation| {
            inputs
                .into_par_iter()
                .map(|input| {
                    check_cancelled(&cancellation)?;
                    Ok(decode_image(input.bytes()).map(Arc::new))
                })
                .collect()
        },
    )
    .await
}

pub(super) fn validate_decoded_cardinality(
    slots: &[OcrProviderBatchSlot],
    decoded: &[SharedDecodedImage],
) -> UseResult<()> {
    if slots.len() != decoded.len() {
        return Err(UseError::new(
            "use.ocr.provider_input_invalid",
            "Document-fast shared image decoding changed slot cardinality.",
        ));
    }
    Ok(())
}
