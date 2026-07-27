//! Standard MCP tools for the built-in OCR domain.

use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{CallToolResult, Implementation, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_handler, tool_router, ServerHandler, ServiceExt};
use serde::Serialize;

use crate::{OcrClient, OcrDiagnostic, OcrRequest, OcrResult, UseError, UseResult};

#[derive(Clone)]
pub struct OcrMcpServer {
    client: OcrClient,
    tool_router: ToolRouter<Self>,
}

impl OcrMcpServer {
    pub fn new(client: OcrClient) -> Self {
        let mut tool_router = Self::tool_router();
        if let Some(route) = tool_router.map.get_mut("ocr_extract") {
            if let Some(annotations) = route.attr.annotations.as_mut() {
                annotations.open_world_hint = Some(client.provider().sends_source_off_device);
            }
        }
        Self {
            client,
            tool_router,
        }
    }

    #[cfg(feature = "ppocr-v6")]
    pub fn from_env() -> UseResult<Self> {
        Ok(Self::new(OcrClient::from_env()?))
    }

    /// Serve standard MCP framing over stdin/stdout until the peer disconnects.
    pub async fn serve_stdio(self) -> UseResult<()> {
        let service = self
            .serve(rmcp::transport::stdio())
            .await
            .map_err(|error| mcp_error("start", error))?;
        service
            .waiting()
            .await
            .map_err(|error| mcp_error("run", error))?;
        Ok(())
    }
}

#[tool_router]
impl OcrMcpServer {
    #[tool(
        name = "ocr_doctor",
        description = "Inspect the configured OCR provider without reading an image",
        output_schema = rmcp::handler::server::tool::cached_schema_for_type::<OcrDiagnostic>(),
        annotations(
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn ocr_doctor(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        Ok(tool_result(Ok(self.client.diagnostic())))
    }

    #[tool(
        name = "ocr_extract",
        description = "Extract text and available layout evidence from one bounded local image through the configured OCR provider",
        output_schema = rmcp::handler::server::tool::cached_schema_for_type::<OcrResult>(),
        annotations(
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn ocr_extract(
        &self,
        Parameters(request): Parameters<OcrRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        Ok(tool_result(self.client.extract(request).await))
    }
}

#[tool_handler]
impl ServerHandler for OcrMcpServer {
    fn get_info(&self) -> ServerInfo {
        let provider = self.client.provider();
        let data_policy = if provider.sends_source_off_device {
            "This provider may send source bytes off device; preserve that policy boundary."
        } else {
            "This provider keeps source bytes on device."
        };
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "a3s-use-ocr".to_string(),
                title: Some("A3S Use OCR".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                website_url: Some("https://github.com/A3S-Lab/OCR".to_string()),
            },
            instructions: Some(format!(
                "Call ocr_doctor first. Use ocr_extract only for a local image path supplied by the task. The configured provider is '{}' using '{}'. {data_policy} Preserve the source SHA-256 and distinguish OCR text from verified source text.",
                provider.id, provider.engine
            )),
            ..Default::default()
        }
    }
}

fn tool_result<T>(result: UseResult<T>) -> CallToolResult
where
    T: Serialize,
{
    match result {
        Ok(output) => match serde_json::to_value(output) {
            Ok(value) => CallToolResult::structured(value),
            Err(error) => tool_error(UseError::new(
                "use.ocr.output_invalid",
                format!("Failed to encode OCR MCP output: {error}"),
            )),
        },
        Err(error) => tool_error(error),
    }
}

fn tool_error(error: UseError) -> CallToolResult {
    CallToolResult::structured_error(serde_json::to_value(error).unwrap_or_else(|_| {
        serde_json::json!({
            "code": "use.error_encoding_failed",
            "message": "Failed to encode A3S Use error."
        })
    }))
}

fn mcp_error(action: &str, error: impl std::fmt::Display) -> UseError {
    UseError::new(
        "use.ocr.mcp_failed",
        format!("Failed to {action} the OCR MCP server: {error}"),
    )
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;
    use crate::{
        OcrInput, OcrProvider, OcrProviderDescriptor, OcrProviderOutput, OcrProviderStatus,
        Readiness,
    };

    struct RemoteFixtureProvider;

    #[async_trait]
    impl OcrProvider for RemoteFixtureProvider {
        fn descriptor(&self) -> OcrProviderDescriptor {
            OcrProviderDescriptor::new("remote-fixture", "fixture-api", true).unwrap()
        }

        fn diagnostic(&self) -> OcrProviderStatus {
            OcrProviderStatus {
                readiness: Readiness::Ready,
                model: None,
                model_dir: None,
                message: "ready".to_string(),
                suggestions: Vec::new(),
            }
        }

        async fn recognize(&self, _input: OcrInput) -> UseResult<OcrProviderOutput> {
            Ok(OcrProviderOutput::default())
        }
    }

    #[test]
    fn server_exposes_typed_annotated_ocr_tools() {
        let client = OcrClient::from_env().unwrap();
        let server = OcrMcpServer::new(client);
        let mut tools = server.tool_router.list_all();
        tools.sort_by(|left, right| left.name.cmp(&right.name));
        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.name.as_ref())
                .collect::<Vec<&str>>(),
            ["ocr_doctor", "ocr_extract"]
        );
        let doctor = tools.iter().find(|tool| tool.name == "ocr_doctor").unwrap();
        let extract = tools
            .iter()
            .find(|tool| tool.name == "ocr_extract")
            .unwrap();
        assert!(doctor.output_schema.is_some());
        assert!(extract.output_schema.is_some());
        assert_eq!(
            doctor
                .annotations
                .as_ref()
                .and_then(|annotations| annotations.open_world_hint),
            Some(false)
        );
        assert_eq!(
            extract
                .annotations
                .as_ref()
                .and_then(|annotations| annotations.open_world_hint),
            Some(false)
        );
    }

    #[test]
    fn server_projects_an_injected_providers_data_boundary() {
        let client = OcrClient::with_provider(RemoteFixtureProvider).unwrap();
        let server = OcrMcpServer::new(client);
        let extract = server
            .tool_router
            .list_all()
            .into_iter()
            .find(|tool| tool.name == "ocr_extract")
            .unwrap();

        assert_eq!(
            extract
                .annotations
                .and_then(|annotations| annotations.open_world_hint),
            Some(true)
        );
    }
}
