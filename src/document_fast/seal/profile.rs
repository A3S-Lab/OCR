//! Reviewed seal-layout model contracts selected only by exact asset identity.

const LARGE_GRAPH: &str = include_str!("graphs/picodet_l_layout_3cls.json");
const SMALL_GRAPH: &str = include_str!("graphs/picodet_s_layout_3cls.json");

// Both reviewed PicoDet 3-class bundles publish the same MultiClassNMS
// contract. Keep these values tied to the released model contracts instead of
// deriving acceptance thresholds from downstream fixtures.
pub(super) const SCORE_THRESHOLD: f32 = 0.3;
pub(super) const NMS_IOU_THRESHOLD: f32 = 0.5;
pub(super) const KEEP_TOP_K: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PicodetLayoutProfile {
    Large,
    Small,
}

impl PicodetLayoutProfile {
    pub(super) const ALL: [Self; 2] = [Self::Large, Self::Small];

    pub(super) fn from_weight_identity(bytes: u64, sha256: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|profile| {
            profile.weights_bytes() == bytes && profile.weights_file_sha256() == sha256
        })
    }

    pub(super) const fn family(self) -> &'static str {
        match self {
            Self::Large => "picodet-l-layout-3cls",
            Self::Small => "picodet-s-layout-3cls",
        }
    }

    pub(super) const fn revision(self) -> &'static str {
        "paddleocr-paddle3-reviewed-v1"
    }

    pub(super) const fn source_graph_sha256(self) -> &'static str {
        match self {
            Self::Large => "9df09659ed993444d068cc41b8b3e69306890b79c2af6f674d4111ab86e845da",
            Self::Small => "d95d338030f9f8de79339fa7c4f99bac02255f48ee7af059cb60b09bd00188a6",
        }
    }

    pub(super) const fn graph_sha256(self) -> &'static str {
        match self {
            Self::Large => "6903f703d263e965d82bd0327f51dceb3f787ffff0b9411960a551f1f8119bd5",
            Self::Small => "6e12e814ca56d64a7da667dcdf62a3bd282b9a55a5899886954201ab431885d2",
        }
    }

    pub(super) const fn weights_file_sha256(self) -> &'static str {
        match self {
            Self::Large => "88c2d62f5ad48ff0487d0dc86e347f45ca369746cb8e0c8693ed9ecf1cb7fc9e",
            Self::Small => "bd9e15a38068fdb4e1999c5b1e51617f146e61f7d6069d5fbee2ac91881d523e",
        }
    }

    pub(super) const fn weights_collection_sha256(self) -> &'static str {
        match self {
            Self::Large => "361452be560223a2a4799026f92bf6ed2612f7a6d9abf8db4679a806e5eab965",
            Self::Small => "de66693095930eb3ba04c9b8478ba58523ad2c7638df406bfb8f9effafe5cd0f",
        }
    }

    pub(super) const fn weights_bytes(self) -> u64 {
        match self {
            Self::Large => 23_361_700,
            Self::Small => 4_810_200,
        }
    }

    pub(super) const fn input_side(self) -> usize {
        match self {
            Self::Large => 640,
            Self::Small => 480,
        }
    }

    pub(super) const fn location_count(self) -> usize {
        match self {
            Self::Large => 8_500,
            Self::Small => 4_789,
        }
    }

    pub(super) const fn output_width(self) -> usize {
        7
    }

    pub(super) const fn image_class_index(self) -> usize {
        0
    }

    pub(super) const fn seal_class_index(self) -> usize {
        2
    }

    pub(super) const fn embedded_graph(self) -> &'static str {
        match self {
            Self::Large => LARGE_GRAPH,
            Self::Small => SMALL_GRAPH,
        }
    }

    #[cfg(test)]
    pub(super) const fn expected_node_count(self) -> usize {
        match self {
            Self::Large => 518,
            Self::Small => 454,
        }
    }

    #[cfg(test)]
    pub(super) const fn expected_initializer_count(self) -> usize {
        match self {
            Self::Large => 588,
            Self::Small => 508,
        }
    }

    pub(super) const fn tensor_elements_per_view(self) -> usize {
        3 * self.input_side() * self.input_side()
    }
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;

    #[test]
    fn every_reviewed_graph_keeps_its_exact_identity_and_inventory() {
        for profile in PicodetLayoutProfile::ALL {
            let graph_source = profile.embedded_graph();
            assert_eq!(
                format!("{:x}", Sha256::digest(graph_source)),
                profile.graph_sha256()
            );
            let graph: serde_json::Value = serde_json::from_str(graph_source).unwrap();
            assert_eq!(graph["family"], profile.family());
            assert_eq!(graph["role"], "layout-raw-head");
            assert_eq!(graph["source"]["sha256"], profile.source_graph_sha256());
            assert_eq!(
                graph["nodes"].as_array().unwrap().len(),
                profile.expected_node_count()
            );
            assert_eq!(
                graph["initializers"].as_array().unwrap().len(),
                profile.expected_initializer_count()
            );
        }
    }

    #[test]
    fn only_exact_weight_identities_select_a_profile() {
        for profile in PicodetLayoutProfile::ALL {
            assert_eq!(
                PicodetLayoutProfile::from_weight_identity(
                    profile.weights_bytes(),
                    profile.weights_file_sha256()
                ),
                Some(profile)
            );
        }
        assert_eq!(
            PicodetLayoutProfile::from_weight_identity(
                PicodetLayoutProfile::Small.weights_bytes(),
                PicodetLayoutProfile::Large.weights_file_sha256()
            ),
            None
        );
    }

    #[test]
    fn host_postprocessing_matches_the_reviewed_model_contract() {
        assert_eq!(SCORE_THRESHOLD, 0.3);
        assert_eq!(NMS_IOU_THRESHOLD, 0.5);
        assert_eq!(KEEP_TOP_K, 100);
        assert_eq!(PicodetLayoutProfile::Large.image_class_index(), 0);
        assert_eq!(PicodetLayoutProfile::Large.seal_class_index(), 2);
        assert_eq!(PicodetLayoutProfile::Small.image_class_index(), 0);
        assert_eq!(PicodetLayoutProfile::Small.seal_class_index(), 2);
    }
}
