use a3s_power::inference::ModelSession;
use a3s_use_core::UseResult;
use tokio_util::sync::CancellationToken;

use super::{
    admission, batching, runtime_error, trace_view_count, PageAccumulator, SealSession,
    SealStageRunner, ViewReference,
};
use crate::OcrExecutionReceipt;

pub(super) async fn run_single(
    runner: &SealStageRunner,
    pages: &mut [PageAccumulator],
    sessions: &[ModelSession<SealSession>],
    cancellation: CancellationToken,
) -> UseResult<Vec<OcrExecutionReceipt>> {
    let (refinement_views, mut receipts) =
        run_initial(runner, pages, sessions, cancellation.clone()).await?;
    run_refinement(
        runner,
        pages,
        sessions,
        refinement_views,
        &mut receipts,
        cancellation.clone(),
    )
    .await?;
    receipts.extend(run_adjacent(runner, pages, sessions, cancellation).await?);
    Ok(receipts)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn run_paired(
    runner: &SealStageRunner,
    first_pages: &mut [PageAccumulator],
    second_pages: &mut [PageAccumulator],
    sessions: &[ModelSession<SealSession>],
    first_session_count: usize,
    second_session_count: usize,
    cancellation: CancellationToken,
) -> UseResult<(Vec<OcrExecutionReceipt>, Vec<OcrExecutionReceipt>)> {
    if first_session_count == 0
        || second_session_count == 0
        || first_session_count.saturating_add(second_session_count) != sessions.len()
    {
        return Err(runtime_error(
            "PicoDet paired layout execution received an invalid session partition.",
        ));
    }
    let first_sessions = &sessions[..first_session_count];
    let second_sessions = &sessions[first_session_count..];
    let first = run_single(runner, first_pages, first_sessions, cancellation.clone());
    let second = run_single(runner, second_pages, second_sessions, cancellation);
    let (first, second) = tokio::join!(first, second);
    Ok((first?, second?))
}

async fn run_initial(
    runner: &SealStageRunner,
    pages: &mut [PageAccumulator],
    sessions: &[ModelSession<SealSession>],
    cancellation: CancellationToken,
) -> UseResult<(Vec<ViewReference>, Vec<OcrExecutionReceipt>)> {
    let views = admission::model_contract_views(pages);
    trace_view_count("model-contract", views.len());
    let mut receipts = Vec::new();
    if views.is_empty() {
        return Ok((Vec::new(), receipts));
    }
    if sessions.is_empty() {
        return Err(runtime_error(
            "PicoDet layout execution requires at least one prepared session.",
        ));
    }
    let refinement_views = batching::run_view_batches(
        sessions,
        runner.assets.profile,
        views,
        pages,
        &mut receipts,
        cancellation,
        true,
    )
    .await?;
    trace_view_count("model-ranked-refinement", refinement_views.len());
    Ok((refinement_views, receipts))
}

async fn run_refinement(
    runner: &SealStageRunner,
    pages: &mut [PageAccumulator],
    sessions: &[ModelSession<SealSession>],
    views: Vec<ViewReference>,
    receipts: &mut Vec<OcrExecutionReceipt>,
    cancellation: CancellationToken,
) -> UseResult<()> {
    if views.is_empty() {
        return Ok(());
    }
    batching::run_view_batches(
        sessions,
        runner.assets.profile,
        views,
        pages,
        receipts,
        cancellation,
        false,
    )
    .await
    .map(|_| ())
}

async fn run_adjacent(
    runner: &SealStageRunner,
    pages: &mut [PageAccumulator],
    sessions: &[ModelSession<SealSession>],
    cancellation: CancellationToken,
) -> UseResult<Vec<OcrExecutionReceipt>> {
    let views = admission::adjacent_boundary_views(pages);
    trace_view_count("adjacent-boundary-verification", views.len());
    let mut receipts = Vec::new();
    if views.is_empty() {
        return Ok(receipts);
    }
    batching::run_view_batches(
        sessions,
        runner.assets.profile,
        views,
        pages,
        &mut receipts,
        cancellation,
        false,
    )
    .await?;
    Ok(receipts)
}
