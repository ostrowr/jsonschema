use crate::{
    compiler,
    error::ValidationError,
    evaluation::ErrorDescription,
    keywords::CompilationResult,
    node::SchemaNode,
    paths::{LazyLocation, Location, RefTracker},
    types::JsonType,
    validator::{EvaluationResult, Validate, ValidationContext},
    InstanceRef,
};
use ahash::AHashMap;
use serde_json::{Map, Value};

pub(crate) struct OneOfValidator {
    schemas: Vec<SchemaNode>,
    location: Location,
    discriminator: Option<OneOfDiscriminator>,
}

struct OneOfDiscriminator {
    property: String,
    branches: AHashMap<String, usize>,
}

impl OneOfValidator {
    #[inline]
    pub(crate) fn compile<'a>(ctx: &compiler::Context, schema: &'a Value) -> CompilationResult<'a> {
        if let Value::Array(items) = schema {
            let ctx = ctx.new_at_location("oneOf");
            let discriminator = compile_discriminator(&ctx, items);
            let mut schemas = Vec::with_capacity(items.len());
            for (idx, item) in items.iter().enumerate() {
                let ctx = ctx.new_at_location(idx);
                let node = compiler::compile(&ctx, ctx.as_resource_ref(item))?;
                schemas.push(node);
            }
            Ok(Box::new(OneOfValidator {
                schemas,
                location: ctx.location().clone(),
                discriminator,
            }))
        } else {
            let location = ctx.location().join("oneOf");
            Err(ValidationError::single_type_error(
                location.clone(),
                location,
                Location::new(),
                schema,
                JsonType::Array,
            ))
        }
    }

    fn get_first_valid(&self, instance: &Value, ctx: &mut ValidationContext) -> Option<usize> {
        let mut first_valid_idx = None;
        for (idx, node) in self.schemas.iter().enumerate() {
            if node.is_valid(instance, ctx) {
                first_valid_idx = Some(idx);
                break;
            }
        }
        first_valid_idx
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn are_others_valid(&self, instance: &Value, idx: usize, ctx: &mut ValidationContext) -> bool {
        self.schemas
            .iter()
            .skip(idx + 1)
            .any(|n| n.is_valid(instance, ctx))
    }

    fn get_first_valid_instance(
        &self,
        instance: InstanceRef<'_>,
        ctx: &mut ValidationContext,
    ) -> Option<usize> {
        self.schemas
            .iter()
            .position(|node| node.is_valid_instance(instance, ctx))
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn are_other_instances_valid(
        &self,
        instance: InstanceRef<'_>,
        index: usize,
        ctx: &mut ValidationContext,
    ) -> bool {
        self.schemas
            .iter()
            .skip(index + 1)
            .any(|node| node.is_valid_instance(instance, ctx))
    }

    fn discriminated_branch(&self, instance: InstanceRef<'_>) -> Option<usize> {
        let discriminator = self.discriminator.as_ref()?;
        let value = instance.as_object()?.get(&discriminator.property)?;
        discriminator.branches.get(value.as_str()?).copied()
    }
}

fn compile_discriminator(ctx: &compiler::Context, schemas: &[Value]) -> Option<OneOfDiscriminator> {
    let first = discriminator_candidates(ctx, schemas.first()?)?;
    for (property, _) in first {
        let mut branches = AHashMap::with_capacity(schemas.len());
        let mut complete = true;
        for (index, schema) in schemas.iter().enumerate() {
            let Some(value) = required_string_const(ctx, schema, &property) else {
                complete = false;
                break;
            };
            if branches.insert(value.clone(), index).is_some() {
                complete = false;
                break;
            }
        }
        if complete {
            return Some(OneOfDiscriminator { property, branches });
        }
    }
    None
}

fn discriminator_candidates(
    ctx: &compiler::Context,
    schema: &Value,
) -> Option<Vec<(String, String)>> {
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        let resolved = ctx.lookup(reference).ok()?;
        return discriminator_candidates_from_schema(resolved.contents());
    }
    discriminator_candidates_from_schema(schema)
}

fn discriminator_candidates_from_schema(schema: &Value) -> Option<Vec<(String, String)>> {
    let required = schema.get("required")?.as_array()?;
    let properties = schema.get("properties")?.as_object()?;
    Some(
        required
            .iter()
            .filter_map(Value::as_str)
            .filter_map(|property| {
                properties
                    .get(property)?
                    .get("const")?
                    .as_str()
                    .map(|value| (property.to_owned(), value.to_owned()))
            })
            .collect(),
    )
}

fn required_string_const(
    ctx: &compiler::Context,
    schema: &Value,
    property: &str,
) -> Option<String> {
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        let resolved = ctx.lookup(reference).ok()?;
        return required_string_const_from_schema(resolved.contents(), property);
    }
    required_string_const_from_schema(schema, property)
}

fn required_string_const_from_schema(schema: &Value, property: &str) -> Option<String> {
    let required = schema.get("required")?.as_array()?;
    if !required
        .iter()
        .any(|value| value.as_str() == Some(property))
    {
        return None;
    }
    schema
        .get("properties")?
        .get(property)?
        .get("const")?
        .as_str()
        .map(str::to_owned)
}

/// Optimized validator for `oneOf` with a single subschema.
/// With exactly one schema, `oneOf` behaves identically to `anyOf`.
pub(crate) struct SingleOneOfValidator {
    node: SchemaNode,
    location: Location,
}

impl SingleOneOfValidator {
    #[inline]
    pub(crate) fn compile<'a>(ctx: &compiler::Context, schema: &'a Value) -> CompilationResult<'a> {
        let one_of_ctx = ctx.new_at_location("oneOf");
        let item_ctx = one_of_ctx.new_at_location(0);
        let node = compiler::compile(&item_ctx, item_ctx.as_resource_ref(schema))?;
        Ok(Box::new(SingleOneOfValidator {
            node,
            location: one_of_ctx.location().clone(),
        }))
    }
}

impl Validate for SingleOneOfValidator {
    fn is_valid(&self, instance: &Value, ctx: &mut ValidationContext) -> bool {
        self.node.is_valid(instance, ctx)
    }

    fn is_valid_instance(&self, instance: InstanceRef<'_>, ctx: &mut ValidationContext) -> bool {
        self.node.is_valid_instance(instance, ctx)
    }

    fn validate<'i>(
        &self,
        instance: &'i Value,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        if self.node.is_valid(instance, ctx) {
            Ok(())
        } else {
            Err(ValidationError::one_of_not_valid(
                self.location.clone(),
                crate::paths::capture_evaluation_path(tracker, &self.location),
                location.into(),
                instance,
                vec![self
                    .node
                    .iter_errors(instance, location, tracker, ctx)
                    .collect()],
            ))
        }
    }

    fn evaluate(
        &self,
        instance: &Value,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        EvaluationResult::from(
            self.node
                .evaluate_instance(instance, location, tracker, ctx),
        )
    }
}

impl Validate for OneOfValidator {
    fn is_valid(&self, instance: &Value, ctx: &mut ValidationContext) -> bool {
        if let Some(index) = self.discriminated_branch(InstanceRef::from_serde(instance)) {
            return self.schemas[index].is_valid(instance, ctx);
        }
        let first_valid_idx = self.get_first_valid(instance, ctx);
        first_valid_idx.is_some_and(|idx| !self.are_others_valid(instance, idx, ctx))
    }

    fn is_valid_instance(&self, instance: InstanceRef<'_>, ctx: &mut ValidationContext) -> bool {
        if let Some(index) = self.discriminated_branch(instance) {
            return self.schemas[index].is_valid_instance(instance, ctx);
        }
        self.get_first_valid_instance(instance, ctx)
            .is_some_and(|index| !self.are_other_instances_valid(instance, index, ctx))
    }

    fn validate<'i>(
        &self,
        instance: &'i Value,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> Result<(), ValidationError<'i>> {
        let first_valid_idx = self.get_first_valid(instance, ctx);
        if let Some(idx) = first_valid_idx {
            if self.are_others_valid(instance, idx, ctx) {
                return Err(ValidationError::one_of_multiple_valid(
                    self.location.clone(),
                    crate::paths::capture_evaluation_path(tracker, &self.location),
                    location.into(),
                    instance,
                    self.schemas
                        .iter()
                        .map(|schema| {
                            schema
                                .iter_errors(instance, location, tracker, ctx)
                                .collect()
                        })
                        .collect(),
                ));
            }
            Ok(())
        } else {
            Err(ValidationError::one_of_not_valid(
                self.location.clone(),
                crate::paths::capture_evaluation_path(tracker, &self.location),
                location.into(),
                instance,
                self.schemas
                    .iter()
                    .map(|schema| {
                        schema
                            .iter_errors(instance, location, tracker, ctx)
                            .collect()
                    })
                    .collect(),
            ))
        }
    }

    fn evaluate(
        &self,
        instance: &Value,
        location: &LazyLocation,
        tracker: Option<&RefTracker>,
        ctx: &mut ValidationContext,
    ) -> EvaluationResult {
        // Use cheap `is_valid` first, then run full `evaluate` only on matching schemas.
        let first_valid_idx = self.get_first_valid(instance, ctx);

        let Some(first_idx) = first_valid_idx else {
            let failures: Vec<_> = self
                .schemas
                .iter()
                .map(|node| node.evaluate_instance(instance, location, tracker, ctx))
                .collect();
            return EvaluationResult::Invalid {
                errors: Vec::new(),
                children: failures,
                annotations: None,
            };
        };

        if self.are_others_valid(instance, first_idx, ctx) {
            let mut successes = Vec::new();
            for (idx, node) in self.schemas.iter().enumerate() {
                if idx == first_idx || node.is_valid(instance, ctx) {
                    let child = node.evaluate_instance(instance, location, tracker, ctx);
                    if child.valid {
                        successes.push(child);
                    }
                }
            }
            EvaluationResult::Invalid {
                errors: vec![ErrorDescription::new(
                    "oneOf",
                    "more than one subschema succeeded".to_string(),
                )],
                children: successes,
                annotations: None,
            }
        } else {
            let child = self.schemas[first_idx].evaluate_instance(instance, location, tracker, ctx);
            EvaluationResult::from(child)
        }
    }
}

#[inline]
pub(crate) fn compile<'a>(
    ctx: &compiler::Context,
    _: &'a Map<String, Value>,
    schema: &'a Value,
) -> Option<CompilationResult<'a>> {
    match schema {
        Value::Array(items) => match items.as_slice() {
            [item] => Some(SingleOneOfValidator::compile(ctx, item)),
            _ => Some(OneOfValidator::compile(ctx, schema)),
        },
        _ => Some(OneOfValidator::compile(ctx, schema)),
    }
}

#[cfg(test)]
mod tests {
    use crate::tests_util;
    use serde_json::{json, Value};
    use test_case::test_case;

    #[test]
    fn required_unique_string_consts_preserve_one_of_semantics() {
        let schema = json!({
            "$defs": {
                "cat": {
                    "type": "object",
                    "required": ["kind", "lives"],
                    "properties": {
                        "kind": {"const": "cat"},
                        "lives": {"type": "integer", "minimum": 1}
                    },
                    "additionalProperties": false
                },
                "dog": {
                    "type": "object",
                    "required": ["kind", "good"],
                    "properties": {
                        "kind": {"const": "dog"},
                        "good": {"type": "boolean"}
                    },
                    "additionalProperties": false
                }
            },
            "oneOf": [
                {"$ref": "#/$defs/cat"},
                {"$ref": "#/$defs/dog"}
            ]
        });
        let validator = crate::validator_for(&schema).unwrap();

        assert!(validator.is_valid(&json!({"kind": "cat", "lives": 9})));
        assert!(validator.is_valid(&json!({"kind": "dog", "good": true})));
        assert!(!validator.is_valid(&json!({"kind": "cat", "lives": 0})));
        assert!(!validator.is_valid(&json!({"kind": "dog", "good": "yes"})));
        assert!(!validator.is_valid(&json!({"kind": "bird"})));
        assert!(!validator.is_valid(&json!({"lives": 9})));
    }

    #[test]
    fn repeated_const_values_still_check_every_one_of_branch() {
        let schema = json!({
            "oneOf": [
                {
                    "required": ["kind"],
                    "properties": {"kind": {"const": "same"}}
                },
                {
                    "required": ["kind"],
                    "properties": {"kind": {"const": "same"}}
                }
            ]
        });
        let validator = crate::validator_for(&schema).unwrap();
        assert!(!validator.is_valid(&json!({"kind": "same"})));
    }

    #[cfg(feature = "jiter")]
    #[test]
    fn borrowed_jiter_instances_use_discriminated_one_of_semantics() {
        let schema = json!({
            "oneOf": [
                {
                    "required": ["kind", "value"],
                    "properties": {
                        "kind": {"const": "text"},
                        "value": {"type": "string"}
                    }
                },
                {
                    "required": ["kind", "value"],
                    "properties": {
                        "kind": {"const": "number"},
                        "value": {"type": "integer"}
                    }
                }
            ]
        });
        let validator = crate::validator_for(&schema).unwrap();
        for (source, expected) in [
            (r#"{"kind":"text","value":"ok"}"#, true),
            (r#"{"kind":"number","value":42}"#, true),
            (r#"{"kind":"number","value":"bad"}"#, false),
            (r#"{"kind":"unknown","value":42}"#, false),
        ] {
            let parsed = jiter::JsonValue::parse(source.as_bytes(), false).unwrap();
            assert_eq!(
                validator.is_valid_instance(crate::InstanceRef::from_jiter(&parsed)),
                expected,
                "{source}"
            );
        }
    }

    #[test_case(&json!({"oneOf": [{"type": "string"}]}), &json!(0), "/oneOf")]
    #[test_case(&json!({"oneOf": [{"type": "string"}, {"maxLength": 3}]}), &json!(""), "/oneOf")]
    fn location(schema: &Value, instance: &Value, expected: &str) {
        tests_util::assert_schema_location(schema, instance, expected);
    }
}
