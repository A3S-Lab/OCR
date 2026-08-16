use serde_json::Value;

use super::RECOGNITION_GRAPH;

#[derive(Clone, Copy)]
enum Activation {
    Relu,
    Gelu,
    GatedHardSigmoid,
}

#[test]
fn recognition_graph_keeps_the_biased_activation_inventory() {
    let graph: Value = serde_json::from_str(RECOGNITION_GRAPH).unwrap();

    assert_eq!(reviewed_windows(&graph, Activation::Relu), 10);
    assert_eq!(reviewed_windows(&graph, Activation::Gelu), 13);
    assert_eq!(reviewed_windows(&graph, Activation::GatedHardSigmoid), 5);
}

#[test]
fn recognition_graph_keeps_the_layer_norm_affine_tail_inventory() {
    let graph: Value = serde_json::from_str(RECOGNITION_GRAPH).unwrap();
    let nodes = graph["nodes"].as_array().unwrap();
    let signature = ["Add", "Sqrt", "Div", "Mul", "Add"];
    let windows = nodes
        .windows(signature.len())
        .filter(|window| {
            window
                .iter()
                .map(|node| node["op"].as_str().unwrap())
                .eq(signature)
        })
        .collect::<Vec<_>>();
    assert_eq!(windows.len(), 5);

    for window in windows {
        assert_layer_norm_affine_tail(&graph, nodes, window);
    }
}

fn reviewed_windows(graph: &Value, activation: Activation) -> usize {
    let nodes = graph["nodes"].as_array().unwrap();
    let signature: &[&str] = match activation {
        Activation::Relu => &["Conv", "Add", "Identity", "Relu"],
        Activation::Gelu => &[
            "Conv", "Add", "Identity", "Identity", "Div", "Erf", "Add", "Mul", "Mul",
        ],
        Activation::GatedHardSigmoid => &["Conv", "Add", "Identity", "HardSigmoid", "Mul"],
    };
    nodes
        .windows(signature.len())
        .filter(|window| {
            window
                .iter()
                .map(|node| node["op"].as_str().unwrap())
                .eq(signature.iter().copied())
        })
        .map(|window| {
            assert_reviewed_window(graph, nodes, window, activation);
            1
        })
        .sum()
}

fn assert_reviewed_window(
    graph: &Value,
    nodes: &[Value],
    window: &[Value],
    activation: Activation,
) {
    let convolution = &window[0];
    let add = &window[1];
    let convolution_inputs = convolution["inputs"].as_array().unwrap();
    assert_eq!(convolution_inputs.len(), 2);
    let convolution_output = output(convolution);
    assert_private(graph, nodes, convolution_output, 1);
    assert_eq!(uses_in(add, convolution_output), 1);

    let bias = add["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|input| input.as_str().unwrap())
        .find(|input| *input != convolution_output)
        .unwrap();
    let weight = convolution_inputs[1].as_str().unwrap();
    assert_channel_bias(graph, weight, bias);

    let identity_count = match activation {
        Activation::Gelu => 2,
        Activation::Relu | Activation::GatedHardSigmoid => 1,
    };
    let mut current = output(add);
    for identity in &window[2..2 + identity_count] {
        assert_private(graph, nodes, current, 1);
        assert_eq!(identity["inputs"][0].as_str(), Some(current));
        current = output(identity);
    }

    match activation {
        Activation::Relu => {
            let relu = &window[2 + identity_count];
            assert_private(graph, nodes, current, 1);
            assert_eq!(relu["inputs"][0].as_str(), Some(current));
        }
        Activation::GatedHardSigmoid => {
            let hard_sigmoid = &window[2 + identity_count];
            let multiply = &window[3 + identity_count];
            assert_private(graph, nodes, current, 1);
            assert_eq!(hard_sigmoid["inputs"][0].as_str(), Some(current));
            let gate = output(hard_sigmoid);
            assert_private(graph, nodes, gate, 1);
            assert_eq!(uses_in(multiply, gate), 1);
        }
        Activation::Gelu => {
            let gelu = &window[2 + identity_count..];
            assert_private(graph, nodes, current, 2);
            assert_eq!(gelu[0]["inputs"][0].as_str(), Some(current));
            let divide = output(&gelu[0]);
            let erf = output(&gelu[1]);
            let add = output(&gelu[2]);
            let multiply = output(&gelu[3]);
            assert_eq!(gelu[1]["inputs"][0].as_str(), Some(divide));
            assert_eq!(uses_in(&gelu[2], erf), 1);
            assert_eq!(uses_in(&gelu[3], current), 1);
            assert_eq!(uses_in(&gelu[3], add), 1);
            assert_eq!(uses_in(&gelu[4], multiply), 1);
            for value in [divide, erf, add, multiply] {
                assert_private(graph, nodes, value, 1);
            }
        }
    }
}

fn assert_channel_bias(graph: &Value, weight: &str, bias: &str) {
    let initializers = graph["initializers"].as_array().unwrap();
    let weight = initializer(initializers, weight);
    let bias = initializer(initializers, bias);
    assert_eq!(weight["dtype"], "float32");
    assert_eq!(bias["dtype"], "float32");
    let output_channels = weight["shape"][0].as_u64().unwrap();
    assert_eq!(
        bias["shape"].as_array().unwrap(),
        &[
            Value::from(1),
            Value::from(output_channels),
            Value::from(1),
            Value::from(1),
        ]
    );
}

fn assert_layer_norm_affine_tail(graph: &Value, nodes: &[Value], window: &[Value]) {
    assert!(window.iter().all(|node| node["attributes"]
        .as_object()
        .is_some_and(serde_json::Map::is_empty)));
    let add_epsilon = &window[0];
    let sqrt = &window[1];
    let divide = &window[2];
    let multiply = &window[3];
    let add_bias = &window[4];
    let epsilon_output = output(add_epsilon);
    let sqrt_output = output(sqrt);
    let divide_output = output(divide);
    let multiply_output = output(multiply);
    assert_eq!(sqrt["inputs"][0].as_str(), Some(epsilon_output));
    assert_eq!(divide["inputs"][1].as_str(), Some(sqrt_output));
    assert_eq!(uses_in(multiply, divide_output), 1);
    assert_eq!(uses_in(add_bias, multiply_output), 1);
    for value in [epsilon_output, sqrt_output, divide_output, multiply_output] {
        assert_private(graph, nodes, value, 1);
    }

    let initializers = graph["initializers"].as_array().unwrap();
    let epsilon = other_input(add_epsilon, epsilon_output, |name| {
        initializers.iter().any(|value| {
            value["name"].as_str() == Some(name)
                && value["dtype"] == "float32"
                && value["shape"] == serde_json::json!([1])
        })
    });
    let variance = add_epsilon["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|input| input.as_str().unwrap())
        .find(|input| *input != epsilon)
        .unwrap();
    assert_ne!(divide["inputs"][0].as_str(), Some(variance));

    let scale = other_input(multiply, divide_output, |_| true);
    let bias = other_input(add_bias, multiply_output, |_| true);
    for affine in [scale, bias] {
        let value = initializer(initializers, affine);
        assert_eq!(value["dtype"], "float32");
        assert_eq!(value["shape"], serde_json::json!([120]));
    }
}

fn other_input<'a>(node: &'a Value, chained: &str, predicate: impl Fn(&str) -> bool) -> &'a str {
    node["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|input| input.as_str().unwrap())
        .find(|input| *input != chained && predicate(input))
        .unwrap()
}

fn initializer<'a>(initializers: &'a [Value], name: &str) -> &'a Value {
    initializers
        .iter()
        .find(|initializer| initializer["name"].as_str() == Some(name))
        .unwrap()
}

fn assert_private(graph: &Value, nodes: &[Value], value: &str, expected_uses: usize) {
    assert_ne!(graph["outputs"][0]["name"].as_str(), Some(value));
    assert_eq!(
        nodes.iter().map(|node| uses_in(node, value)).sum::<usize>(),
        expected_uses
    );
}

fn uses_in(node: &Value, value: &str) -> usize {
    node["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|input| input.as_str() == Some(value))
        .count()
}

fn output(node: &Value) -> &str {
    node["outputs"][0].as_str().unwrap()
}
