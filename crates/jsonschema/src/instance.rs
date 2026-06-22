//! Borrowed JSON instance views used by the flag-validation hot path.
//!
//! Schema compilation and detailed error reporting continue to use
//! [`serde_json::Value`]. [`InstanceRef`] lets the already-compiled validator
//! inspect other in-memory JSON representations without allocating a second
//! value tree.

#![cfg_attr(feature = "python", allow(unsafe_code))]

use std::iter::FusedIterator;

use num_cmp::NumCmp;
use serde_json::{Map, Number, Value};

#[cfg(feature = "python")]
use pyo3::{
    ffi,
    prelude::*,
    types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
    Borrowed,
};

/// A borrowed JSON value accepted by flag validation.
#[derive(Clone, Copy, Debug)]
pub struct InstanceRef<'a> {
    repr: InstanceRepr<'a>,
}

#[derive(Clone, Copy, Debug)]
enum InstanceRepr<'a> {
    Serde(&'a Value),
    #[cfg(feature = "jiter")]
    Jiter(&'a jiter::JsonValue<'a>),
    #[cfg(feature = "python")]
    Python(PythonValue<'a>),
}

#[cfg(feature = "python")]
#[derive(Clone, Copy)]
pub struct PythonValue<'a> {
    object: std::ptr::NonNull<ffi::PyObject>,
    py: Python<'a>,
}

#[cfg(feature = "python")]
impl std::fmt::Debug for PythonValue<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("PythonValue")
            .field(&self.object)
            .finish()
    }
}

#[cfg(feature = "python")]
impl<'a> PythonValue<'a> {
    fn from_bound<'py>(value: &'a Bound<'py, PyAny>) -> Self
    where
        'py: 'a,
    {
        Self {
            object: std::ptr::NonNull::new(value.as_ptr())
                .expect("Python objects are never represented by null pointers"),
            py: value.py(),
        }
    }

    unsafe fn from_ptr(py: Python<'a>, value: *mut ffi::PyObject) -> Self {
        Self {
            object: std::ptr::NonNull::new(value)
                .expect("borrowed Python container entries are never null"),
            py,
        }
    }

    fn borrowed(self) -> Borrowed<'a, 'a, PyAny> {
        // SAFETY: `object` remains owned by the Python value graph borrowed for
        // `'a`, and `py` proves the interpreter lock is held for that lifetime.
        unsafe { Borrowed::from_ptr(self.py, self.object.as_ptr()) }
    }
}

impl<'a> InstanceRef<'a> {
    /// Borrow a `serde_json` value.
    #[must_use]
    pub const fn from_serde(value: &'a Value) -> Self {
        Self {
            repr: InstanceRepr::Serde(value),
        }
    }

    /// Borrow a jiter value.
    #[cfg(feature = "jiter")]
    #[must_use]
    pub const fn from_jiter(value: &'a jiter::JsonValue<'a>) -> Self {
        Self {
            repr: InstanceRepr::Jiter(value),
        }
    }

    /// Borrow a Python JSON value while the interpreter lock is held.
    #[cfg(feature = "python")]
    #[must_use]
    pub fn from_python<'py>(value: &'a Bound<'py, PyAny>) -> Self
    where
        'py: 'a,
    {
        Self {
            repr: InstanceRepr::Python(PythonValue::from_bound(value)),
        }
    }

    #[cfg(feature = "python")]
    fn from_python_value(value: PythonValue<'a>) -> Self {
        Self {
            repr: InstanceRepr::Python(value),
        }
    }

    /// Return the underlying serde value when this view was created from one.
    #[must_use]
    pub const fn as_serde(self) -> Option<&'a Value> {
        match self.repr {
            InstanceRepr::Serde(value) => Some(value),
            #[cfg(feature = "jiter")]
            InstanceRepr::Jiter(_) => None,
            #[cfg(feature = "python")]
            InstanceRepr::Python(_) => None,
        }
    }

    #[must_use]
    pub fn is_null(self) -> bool {
        match self.repr {
            InstanceRepr::Serde(value) => value.is_null(),
            #[cfg(feature = "jiter")]
            InstanceRepr::Jiter(value) => matches!(value, jiter::JsonValue::Null),
            #[cfg(feature = "python")]
            InstanceRepr::Python(value) => value.borrowed().is_none(),
        }
    }

    #[must_use]
    pub fn is_boolean(self) -> bool {
        self.as_bool().is_some()
    }

    #[must_use]
    pub fn as_bool(self) -> Option<bool> {
        match self.repr {
            InstanceRepr::Serde(value) => value.as_bool(),
            #[cfg(feature = "jiter")]
            InstanceRepr::Jiter(value) => match value {
                jiter::JsonValue::Bool(value) => Some(*value),
                _ => None,
            },
            #[cfg(feature = "python")]
            InstanceRepr::Python(value) => value
                .borrowed()
                .cast::<PyBool>()
                .ok()
                .map(|value| value.is_true()),
        }
    }

    #[must_use]
    pub fn is_string(self) -> bool {
        self.as_str().is_some()
    }

    #[must_use]
    pub fn as_str(self) -> Option<&'a str> {
        match self.repr {
            InstanceRepr::Serde(value) => value.as_str(),
            #[cfg(feature = "jiter")]
            InstanceRepr::Jiter(value) => match value {
                jiter::JsonValue::Str(value) => Some(value.as_ref()),
                _ => None,
            },
            #[cfg(feature = "python")]
            InstanceRepr::Python(value) => python_string(value),
        }
    }

    #[must_use]
    pub fn is_number(self) -> bool {
        self.as_number().is_some()
    }

    #[must_use]
    pub fn as_number(self) -> Option<NumberRef<'a>> {
        match self.repr {
            InstanceRepr::Serde(value) => match value {
                Value::Number(value) => Some(NumberRef::Serde(value)),
                _ => None,
            },
            #[cfg(feature = "jiter")]
            InstanceRepr::Jiter(value) => match value {
                jiter::JsonValue::Int(value) => Some(NumberRef::Integer(*value)),
                jiter::JsonValue::BigInt(value) => Some(NumberRef::BigInteger(value)),
                jiter::JsonValue::Float(value) => Some(NumberRef::Float(*value)),
                _ => None,
            },
            #[cfg(feature = "python")]
            InstanceRepr::Python(value) => {
                let borrowed = value.borrowed();
                if borrowed.cast::<PyBool>().is_ok() {
                    None
                } else if borrowed.cast::<PyInt>().is_ok() {
                    Some(NumberRef::PythonInteger(value))
                } else {
                    borrowed
                        .cast::<PyFloat>()
                        .ok()
                        .and_then(|value| value.value().is_finite().then(|| value.value()))
                        .map(NumberRef::Float)
                }
            }
        }
    }

    #[must_use]
    pub fn as_i64(self) -> Option<i64> {
        match self.as_number() {
            Some(number) => number.as_i64(),
            None => None,
        }
    }

    #[must_use]
    pub fn as_u64(self) -> Option<u64> {
        match self.as_number() {
            Some(number) => number.as_u64(),
            None => None,
        }
    }

    #[must_use]
    pub fn as_f64(self) -> Option<f64> {
        self.as_number().and_then(NumberRef::as_f64)
    }

    #[must_use]
    pub fn is_array(self) -> bool {
        self.as_array().is_some()
    }

    #[must_use]
    pub fn as_array(self) -> Option<ArrayRef<'a>> {
        match self.repr {
            InstanceRepr::Serde(value) => match value {
                Value::Array(items) => Some(ArrayRef::Serde(items)),
                _ => None,
            },
            #[cfg(feature = "jiter")]
            InstanceRepr::Jiter(value) => match value {
                jiter::JsonValue::Array(items) => Some(ArrayRef::Jiter(items)),
                _ => None,
            },
            #[cfg(feature = "python")]
            InstanceRepr::Python(value) => {
                let borrowed = value.borrowed();
                if borrowed.cast::<PyList>().is_ok() {
                    Some(ArrayRef::PythonList(value))
                } else if borrowed.cast::<PyTuple>().is_ok() {
                    Some(ArrayRef::PythonTuple(value))
                } else {
                    None
                }
            }
        }
    }

    #[must_use]
    pub fn is_object(self) -> bool {
        self.as_object().is_some()
    }

    #[must_use]
    pub fn as_object(self) -> Option<ObjectRef<'a>> {
        match self.repr {
            InstanceRepr::Serde(value) => match value {
                Value::Object(properties) => Some(ObjectRef::Serde(properties)),
                _ => None,
            },
            #[cfg(feature = "jiter")]
            InstanceRepr::Jiter(value) => match value {
                jiter::JsonValue::Object(properties) => Some(ObjectRef::Jiter(properties)),
                _ => None,
            },
            #[cfg(feature = "python")]
            InstanceRepr::Python(value) => value
                .borrowed()
                .cast::<PyDict>()
                .is_ok()
                .then_some(ObjectRef::Python(value)),
        }
    }

    /// Stable identity of this node for recursive-schema cycle detection.
    #[must_use]
    pub(crate) fn identity(self) -> usize {
        match self.repr {
            InstanceRepr::Serde(value) => std::ptr::from_ref(value).cast::<()>() as usize,
            #[cfg(feature = "jiter")]
            InstanceRepr::Jiter(value) => std::ptr::from_ref(value).cast::<()>() as usize,
            #[cfg(feature = "python")]
            InstanceRepr::Python(value) => value.object.as_ptr().cast::<()>() as usize,
        }
    }

    /// Materialize this view as a serde value.
    #[must_use]
    pub fn to_owned(self) -> Value {
        match self.repr {
            InstanceRepr::Serde(value) => value.clone(),
            #[cfg(feature = "jiter")]
            InstanceRepr::Jiter(value) => jiter_to_serde(value),
            #[cfg(feature = "python")]
            InstanceRepr::Python(value) => python_to_serde(value),
        }
    }

    /// Return whether this view represents a value in the JSON data model.
    ///
    /// Serde and jiter values are JSON by construction. Python values require
    /// a recursive check for unsupported objects, non-string mapping keys,
    /// non-finite numbers, and container cycles.
    #[must_use]
    pub fn is_json(self) -> bool {
        #[cfg(feature = "python")]
        if matches!(self.repr, InstanceRepr::Python(_)) {
            return python_value_is_json(self, &mut std::collections::HashSet::new());
        }
        true
    }

    /// Compare this instance with a serde value using JSON Schema equality.
    #[must_use]
    pub fn equals(self, expected: &Value) -> bool {
        match expected {
            Value::Null => self.is_null(),
            Value::Bool(expected) => self.as_bool().is_some_and(|value| value == *expected),
            Value::Number(expected) => self
                .as_number()
                .is_some_and(|value| value.equals_serde(expected)),
            Value::String(expected) => self.as_str().is_some_and(|value| value == expected),
            Value::Array(expected) => self.as_array().is_some_and(|value| {
                value.len() == expected.len()
                    && value
                        .iter()
                        .zip(expected)
                        .all(|(value, expected)| value.equals(expected))
            }),
            Value::Object(expected) => self.as_object().is_some_and(|value| {
                value.len() == expected.len()
                    && expected.iter().all(|(key, expected)| {
                        value.get(key).is_some_and(|value| value.equals(expected))
                    })
            }),
        }
    }

    #[must_use]
    pub(crate) fn equals_array(self, expected: &[Value]) -> bool {
        self.as_array().is_some_and(|value| {
            value.len() == expected.len()
                && value
                    .iter()
                    .zip(expected)
                    .all(|(value, expected)| value.equals(expected))
        })
    }

    #[must_use]
    pub(crate) fn equals_object(self, expected: &Map<String, Value>) -> bool {
        self.as_object().is_some_and(|value| {
            value.len() == expected.len()
                && expected.iter().all(|(key, expected)| {
                    value.get(key).is_some_and(|value| value.equals(expected))
                })
        })
    }
}

impl<'a> From<&'a Value> for InstanceRef<'a> {
    fn from(value: &'a Value) -> Self {
        Self::from_serde(value)
    }
}

/// A borrowed JSON number.
#[derive(Clone, Copy, Debug)]
pub enum NumberRef<'a> {
    Serde(&'a Number),
    Integer(i64),
    #[cfg(feature = "jiter")]
    BigInteger(&'a num_bigint::BigInt),
    #[cfg(feature = "python")]
    PythonInteger(PythonValue<'a>),
    Float(f64),
}

impl NumberRef<'_> {
    #[must_use]
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Self::Serde(value) => value.as_i64(),
            Self::Integer(value) => Some(value),
            #[cfg(feature = "jiter")]
            Self::BigInteger(_) | Self::Float(_) => None,
            #[cfg(all(not(feature = "jiter"), not(feature = "python")))]
            Self::Float(_) => None,
            #[cfg(feature = "python")]
            Self::PythonInteger(value) => value.borrowed().extract::<i64>().ok(),
            #[cfg(all(feature = "python", not(feature = "jiter")))]
            Self::Float(_) => None,
        }
    }

    #[must_use]
    pub fn as_u64(self) -> Option<u64> {
        match self {
            Self::Serde(value) => value.as_u64(),
            Self::Integer(value) => u64::try_from(value).ok(),
            #[cfg(feature = "jiter")]
            Self::BigInteger(_) | Self::Float(_) => None,
            #[cfg(all(not(feature = "jiter"), not(feature = "python")))]
            Self::Float(_) => None,
            #[cfg(feature = "python")]
            Self::PythonInteger(value) => value.borrowed().extract::<u64>().ok(),
            #[cfg(all(feature = "python", not(feature = "jiter")))]
            Self::Float(_) => None,
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(self) -> Option<f64> {
        match self {
            Self::Serde(value) => value.as_f64(),
            Self::Integer(value) => Some(value as f64),
            #[cfg(feature = "jiter")]
            Self::BigInteger(value) => {
                use num_traits::ToPrimitive;
                value.to_f64()
            }
            #[cfg(feature = "python")]
            Self::PythonInteger(value) => value.borrowed().extract::<f64>().ok(),
            Self::Float(value) => Some(value),
        }
    }

    #[must_use]
    pub fn is_integer(self) -> bool {
        match self {
            Self::Serde(value) => {
                value.is_i64()
                    || value.is_u64()
                    || value.as_f64().is_some_and(|value| value.fract() == 0.0)
            }
            Self::Integer(_) => true,
            #[cfg(feature = "jiter")]
            Self::BigInteger(_) => true,
            #[cfg(feature = "python")]
            Self::PythonInteger(_) => true,
            Self::Float(value) => value.fract() == 0.0,
        }
    }

    #[must_use]
    pub(crate) fn equals_serde(self, expected: &Number) -> bool {
        match self {
            Self::Serde(value) => crate::ext::cmp::equal_numbers(value, expected),
            Self::Integer(value) => number_equals_i64(expected, value),
            #[cfg(feature = "jiter")]
            Self::BigInteger(value) => bigint_equals_serde(value, expected),
            #[cfg(feature = "python")]
            Self::PythonInteger(value) => {
                if let Ok(value) = value.borrowed().extract::<i64>() {
                    number_equals_i64(expected, value)
                } else if let Ok(value) = value.borrowed().extract::<u64>() {
                    if let Some(expected) = expected.as_u64() {
                        value == expected
                    } else if let Some(expected) = expected.as_i64() {
                        NumCmp::num_eq(value, expected)
                    } else {
                        expected
                            .as_f64()
                            .is_some_and(|expected| NumCmp::num_eq(value, expected))
                    }
                } else {
                    python_integer_to_bigint(value)
                        .is_some_and(|value| bigint_equals_serde(&value, expected))
                }
            }
            Self::Float(value) => number_equals_f64(expected, value),
        }
    }
}

fn number_equals_i64(expected: &Number, value: i64) -> bool {
    if let Some(expected) = expected.as_i64() {
        value == expected
    } else if let Some(expected) = expected.as_u64() {
        NumCmp::num_eq(value, expected)
    } else {
        expected
            .as_f64()
            .is_some_and(|expected| NumCmp::num_eq(value, expected))
    }
}

#[allow(clippy::float_cmp)]
fn number_equals_f64(expected: &Number, value: f64) -> bool {
    if let Some(expected) = expected.as_i64() {
        NumCmp::num_eq(value, expected)
    } else if let Some(expected) = expected.as_u64() {
        NumCmp::num_eq(value, expected)
    } else {
        expected.as_f64().is_some_and(|expected| value == expected)
    }
}

#[cfg(any(feature = "jiter", feature = "python"))]
fn bigint_equals_serde(value: &num_bigint::BigInt, expected: &Number) -> bool {
    if let Some(expected) = expected.as_i64() {
        value == &num_bigint::BigInt::from(expected)
    } else if let Some(expected) = expected.as_u64() {
        value == &num_bigint::BigInt::from(expected)
    } else if let Some(expected) = expected.as_f64() {
        use num_traits::FromPrimitive;
        num_bigint::BigInt::from_f64(expected).is_some_and(|expected| value == &expected)
    } else {
        false
    }
}

/// A borrowed JSON array.
#[derive(Clone, Copy, Debug)]
pub enum ArrayRef<'a> {
    Serde(&'a [Value]),
    #[cfg(feature = "jiter")]
    Jiter(&'a [jiter::JsonValue<'a>]),
    #[cfg(feature = "python")]
    PythonList(PythonValue<'a>),
    #[cfg(feature = "python")]
    PythonTuple(PythonValue<'a>),
}

impl<'a> ArrayRef<'a> {
    /// Returns the number of items in this array.
    ///
    /// # Panics
    ///
    /// Panics only if `CPython` reports a negative length for an exact `list` or
    /// `tuple`, which would violate the `CPython` C API contract.
    #[must_use]
    pub fn len(self) -> usize {
        match self {
            Self::Serde(items) => items.len(),
            #[cfg(feature = "jiter")]
            Self::Jiter(items) => items.len(),
            #[cfg(feature = "python")]
            Self::PythonList(value) => {
                // SAFETY: the variant is constructed only after an exact list cast.
                usize::try_from(unsafe { ffi::PyList_Size(value.object.as_ptr()) })
                    .expect("Python list lengths are non-negative")
            }
            #[cfg(feature = "python")]
            Self::PythonTuple(value) => {
                // SAFETY: the variant is constructed only after an exact tuple cast.
                usize::try_from(unsafe { ffi::PyTuple_Size(value.object.as_ptr()) })
                    .expect("Python tuple lengths are non-negative")
            }
        }
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn get(self, index: usize) -> Option<InstanceRef<'a>> {
        match self {
            Self::Serde(items) => items.get(index).map(InstanceRef::from_serde),
            #[cfg(feature = "jiter")]
            Self::Jiter(items) => items.get(index).map(InstanceRef::from_jiter),
            #[cfg(feature = "python")]
            Self::PythonList(value) => {
                if index >= self.len() {
                    return None;
                }
                // SAFETY: bounds were checked and PyList_GetItem returns a
                // borrowed non-null entry owned by the list.
                let index = ffi::Py_ssize_t::try_from(index).ok()?;
                let item = unsafe { ffi::PyList_GetItem(value.object.as_ptr(), index) };
                // SAFETY: see above; the list keeps the child alive for `'a`.
                Some(InstanceRef::from_python_value(unsafe {
                    PythonValue::from_ptr(value.py, item)
                }))
            }
            #[cfg(feature = "python")]
            Self::PythonTuple(value) => {
                if index >= self.len() {
                    return None;
                }
                // SAFETY: bounds were checked and PyTuple_GetItem returns a
                // borrowed non-null entry owned by the tuple.
                let index = ffi::Py_ssize_t::try_from(index).ok()?;
                let item = unsafe { ffi::PyTuple_GetItem(value.object.as_ptr(), index) };
                // SAFETY: see above; the tuple keeps the child alive for `'a`.
                Some(InstanceRef::from_python_value(unsafe {
                    PythonValue::from_ptr(value.py, item)
                }))
            }
        }
    }

    #[must_use]
    pub fn iter(self) -> ArrayIter<'a> {
        match self {
            Self::Serde(items) => ArrayIter::Serde(items.iter()),
            #[cfg(feature = "jiter")]
            Self::Jiter(items) => ArrayIter::Jiter(items.iter()),
            #[cfg(feature = "python")]
            Self::PythonList(value) => ArrayIter::Python {
                value,
                index: 0,
                len: self.len(),
                tuple: false,
            },
            #[cfg(feature = "python")]
            Self::PythonTuple(value) => ArrayIter::Python {
                value,
                index: 0,
                len: self.len(),
                tuple: true,
            },
        }
    }
}

impl<'a> IntoIterator for ArrayRef<'a> {
    type Item = InstanceRef<'a>;
    type IntoIter = ArrayIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub enum ArrayIter<'a> {
    Serde(std::slice::Iter<'a, Value>),
    #[cfg(feature = "jiter")]
    Jiter(std::slice::Iter<'a, jiter::JsonValue<'a>>),
    #[cfg(feature = "python")]
    Python {
        value: PythonValue<'a>,
        index: usize,
        len: usize,
        tuple: bool,
    },
}

impl<'a> Iterator for ArrayIter<'a> {
    type Item = InstanceRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Serde(items) => items.next().map(InstanceRef::from_serde),
            #[cfg(feature = "jiter")]
            Self::Jiter(items) => items.next().map(InstanceRef::from_jiter),
            #[cfg(feature = "python")]
            Self::Python {
                value,
                index,
                len,
                tuple,
            } => {
                if *index >= *len {
                    return None;
                }
                // SAFETY: the index is in bounds and the variant records the
                // concrete container type checked at construction.
                let python_index = ffi::Py_ssize_t::try_from(*index)
                    .expect("an in-bounds Python sequence index fits Py_ssize_t");
                let item = unsafe {
                    if *tuple {
                        ffi::PyTuple_GetItem(value.object.as_ptr(), python_index)
                    } else {
                        ffi::PyList_GetItem(value.object.as_ptr(), python_index)
                    }
                };
                *index += 1;
                // SAFETY: the container owns this borrowed child for `'a`.
                Some(InstanceRef::from_python_value(unsafe {
                    PythonValue::from_ptr(value.py, item)
                }))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Serde(items) => items.size_hint(),
            #[cfg(feature = "jiter")]
            Self::Jiter(items) => items.size_hint(),
            #[cfg(feature = "python")]
            Self::Python { index, len, .. } => {
                let remaining = len.saturating_sub(*index);
                (remaining, Some(remaining))
            }
        }
    }
}

impl ExactSizeIterator for ArrayIter<'_> {}
impl FusedIterator for ArrayIter<'_> {}

/// A borrowed JSON object.
#[derive(Clone, Copy, Debug)]
pub enum ObjectRef<'a> {
    Serde(&'a Map<String, Value>),
    #[cfg(feature = "jiter")]
    Jiter(&'a [(std::borrow::Cow<'a, str>, jiter::JsonValue<'a>)]),
    #[cfg(feature = "python")]
    Python(PythonValue<'a>),
}

impl<'a> ObjectRef<'a> {
    /// Returns the number of properties in this object.
    ///
    /// # Panics
    ///
    /// Panics only if `CPython` reports a negative length for an exact `dict`,
    /// which would violate the `CPython` C API contract.
    #[must_use]
    pub fn len(self) -> usize {
        match self {
            Self::Serde(properties) => properties.len(),
            #[cfg(feature = "jiter")]
            Self::Jiter(properties) => properties.len(),
            #[cfg(feature = "python")]
            Self::Python(value) => {
                // SAFETY: the variant is created only after a dict cast.
                usize::try_from(unsafe { ffi::PyDict_Size(value.object.as_ptr()) })
                    .expect("Python dict lengths are non-negative")
            }
        }
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn get(self, key: &str) -> Option<InstanceRef<'a>> {
        match self {
            Self::Serde(properties) => properties.get(key).map(InstanceRef::from_serde),
            #[cfg(feature = "jiter")]
            Self::Jiter(properties) => properties
                .iter()
                .find(|(candidate, _)| candidate == key)
                .map(|(_, value)| InstanceRef::from_jiter(value)),
            #[cfg(feature = "python")]
            Self::Python(_) => self
                .iter()
                .find_map(|(candidate, value)| (candidate == key).then_some(value)),
        }
    }

    #[must_use]
    pub fn contains_key(self, key: &str) -> bool {
        self.get(key).is_some()
    }

    #[must_use]
    pub fn iter(self) -> ObjectIter<'a> {
        match self {
            Self::Serde(properties) => ObjectIter::Serde(properties.iter()),
            #[cfg(feature = "jiter")]
            Self::Jiter(properties) => ObjectIter::Jiter(properties.iter()),
            #[cfg(feature = "python")]
            Self::Python(value) => ObjectIter::Python {
                value,
                position: 0,
                remaining: self.len(),
            },
        }
    }

    #[must_use]
    pub fn keys(self) -> ObjectKeys<'a> {
        ObjectKeys(self.iter())
    }

    #[must_use]
    pub fn values(self) -> ObjectValues<'a> {
        ObjectValues(self.iter())
    }
}

impl<'a> IntoIterator for ObjectRef<'a> {
    type Item = (&'a str, InstanceRef<'a>);
    type IntoIter = ObjectIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub enum ObjectIter<'a> {
    Serde(serde_json::map::Iter<'a>),
    #[cfg(feature = "jiter")]
    Jiter(std::slice::Iter<'a, (std::borrow::Cow<'a, str>, jiter::JsonValue<'a>)>),
    #[cfg(feature = "python")]
    Python {
        value: PythonValue<'a>,
        position: ffi::Py_ssize_t,
        remaining: usize,
    },
}

impl<'a> Iterator for ObjectIter<'a> {
    type Item = (&'a str, InstanceRef<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Serde(properties) => properties
                .next()
                .map(|(key, value)| (key.as_str(), InstanceRef::from_serde(value))),
            #[cfg(feature = "jiter")]
            Self::Jiter(properties) => properties
                .next()
                .map(|(key, value)| (key.as_ref(), InstanceRef::from_jiter(value))),
            #[cfg(feature = "python")]
            Self::Python {
                value,
                position,
                remaining,
            } => {
                let mut key = std::ptr::null_mut();
                let mut child = std::ptr::null_mut();
                // SAFETY: the variant guarantees a valid dict pointer and
                // PyDict_Next initializes both borrowed output pointers on success.
                let present = unsafe {
                    ffi::PyDict_Next(
                        value.object.as_ptr(),
                        position,
                        &raw mut key,
                        &raw mut child,
                    )
                };
                if present == 0 {
                    *remaining = 0;
                    return None;
                }
                *remaining = remaining.saturating_sub(1);
                // SAFETY: successful PyDict_Next returns non-null borrowed
                // references kept alive by the dict for `'a`.
                let key = unsafe { PythonValue::from_ptr(value.py, key) };
                let child = unsafe { PythonValue::from_ptr(value.py, child) };
                Some((
                    python_string(key).unwrap_or(""),
                    InstanceRef::from_python_value(child),
                ))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Serde(properties) => properties.size_hint(),
            #[cfg(feature = "jiter")]
            Self::Jiter(properties) => properties.size_hint(),
            #[cfg(feature = "python")]
            Self::Python { remaining, .. } => (*remaining, Some(*remaining)),
        }
    }
}

impl ExactSizeIterator for ObjectIter<'_> {}
impl FusedIterator for ObjectIter<'_> {}

pub struct ObjectKeys<'a>(ObjectIter<'a>);

impl<'a> Iterator for ObjectKeys<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(key, _)| key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for ObjectKeys<'_> {}
impl FusedIterator for ObjectKeys<'_> {}

pub struct ObjectValues<'a>(ObjectIter<'a>);

impl<'a> Iterator for ObjectValues<'a> {
    type Item = InstanceRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, value)| value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for ObjectValues<'_> {}
impl FusedIterator for ObjectValues<'_> {}

#[cfg(feature = "python")]
fn python_value_is_json(
    instance: InstanceRef<'_>,
    active_containers: &mut std::collections::HashSet<usize>,
) -> bool {
    if instance.is_null() || instance.is_boolean() || instance.is_string() || instance.is_number() {
        return true;
    }
    if let Some(items) = instance.as_array() {
        let identity = instance.identity();
        if !active_containers.insert(identity) {
            return false;
        }
        let valid = items
            .iter()
            .all(|item| python_value_is_json(item, active_containers));
        active_containers.remove(&identity);
        return valid;
    }
    let InstanceRepr::Python(value) = instance.repr else {
        return false;
    };
    if value.borrowed().cast::<PyDict>().is_err() {
        return false;
    }
    let identity = instance.identity();
    if !active_containers.insert(identity) {
        return false;
    }
    let mut position = 0;
    let mut key = std::ptr::null_mut();
    let mut child = std::ptr::null_mut();
    loop {
        // SAFETY: `value` is a dict and PyDict_Next initializes borrowed
        // pointers whenever it returns non-zero.
        let present = unsafe {
            ffi::PyDict_Next(
                value.object.as_ptr(),
                &raw mut position,
                &raw mut key,
                &raw mut child,
            )
        };
        if present == 0 {
            break;
        }
        // SAFETY: successful PyDict_Next returns non-null values owned by the dict.
        let key = unsafe { PythonValue::from_ptr(value.py, key) };
        if key.borrowed().cast::<PyString>().is_err() {
            active_containers.remove(&identity);
            return false;
        }
        // SAFETY: successful PyDict_Next returns a non-null value owned by the dict.
        let child =
            InstanceRef::from_python_value(unsafe { PythonValue::from_ptr(value.py, child) });
        if !python_value_is_json(child, active_containers) {
            active_containers.remove(&identity);
            return false;
        }
    }
    active_containers.remove(&identity);
    true
}

#[cfg(feature = "python")]
fn python_string(value: PythonValue<'_>) -> Option<&str> {
    value.borrowed().cast::<PyString>().ok()?;
    let mut size = 0;
    // SAFETY: the cast above proves this is a Unicode object. CPython keeps the
    // UTF-8 cache alive for the lifetime of the object borrowed by `value`.
    let data = unsafe { ffi::PyUnicode_AsUTF8AndSize(value.object.as_ptr(), &raw mut size) };
    if data.is_null() {
        return None;
    }
    // SAFETY: CPython returned a valid UTF-8 buffer of `size` bytes.
    let size = usize::try_from(size).expect("Python Unicode lengths are non-negative");
    let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), size) };
    // SAFETY: PyUnicode_AsUTF8AndSize guarantees valid UTF-8.
    Some(unsafe { std::str::from_utf8_unchecked(bytes) })
}

#[cfg(feature = "python")]
fn python_integer_to_bigint(value: PythonValue<'_>) -> Option<num_bigint::BigInt> {
    value.borrowed().str().ok()?.to_str().ok()?.parse().ok()
}

#[cfg(feature = "python")]
fn python_to_serde(value: PythonValue<'_>) -> Value {
    let instance = InstanceRef::from_python_value(value);
    if instance.is_null() {
        Value::Null
    } else if let Some(value) = instance.as_bool() {
        Value::Bool(value)
    } else if let Some(value) = instance.as_str() {
        Value::String(value.to_owned())
    } else if let Some(number) = instance.as_number() {
        match number {
            NumberRef::PythonInteger(value) => {
                if let Ok(value) = value.borrowed().extract::<i64>() {
                    Value::Number(Number::from(value))
                } else if let Ok(value) = value.borrowed().extract::<u64>() {
                    Value::Number(Number::from(value))
                } else {
                    let rendered = value
                        .borrowed()
                        .str()
                        .expect("Python integers always have a string representation");
                    serde_json::from_str(
                        rendered
                            .to_str()
                            .expect("Python integer strings are always valid UTF-8"),
                    )
                    .expect("Python integer strings are valid JSON numbers")
                }
            }
            NumberRef::Float(value) => Number::from_f64(value).map_or(Value::Null, Value::Number),
            _ => unreachable!("Python values produce only PythonInteger or Float number views"),
        }
    } else if let Some(items) = instance.as_array() {
        Value::Array(items.iter().map(InstanceRef::to_owned).collect())
    } else if let Some(properties) = instance.as_object() {
        Value::Object(
            properties
                .iter()
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
                .collect(),
        )
    } else {
        Value::Null
    }
}

#[cfg(feature = "jiter")]
fn jiter_to_serde(value: &jiter::JsonValue<'_>) -> Value {
    match value {
        jiter::JsonValue::Null => Value::Null,
        jiter::JsonValue::Bool(value) => Value::Bool(*value),
        jiter::JsonValue::Int(value) => Value::Number(Number::from(*value)),
        jiter::JsonValue::BigInt(value) => serde_json::from_str(&value.to_string())
            .expect("jiter BigInt is always a syntactically valid JSON number"),
        jiter::JsonValue::Float(value) => {
            Value::Number(Number::from_f64(*value).expect("jiter JSON numbers are always finite"))
        }
        jiter::JsonValue::Str(value) => Value::String(value.to_string()),
        jiter::JsonValue::Array(items) => Value::Array(items.iter().map(jiter_to_serde).collect()),
        jiter::JsonValue::Object(properties) => Value::Object(
            properties
                .iter()
                .map(|(key, value)| (key.to_string(), jiter_to_serde(value)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "python")]
    use pyo3::types::PyAnyMethods;
    use serde_json::json;
    #[cfg(any(feature = "jiter", feature = "python"))]
    use serde_json::Value;

    use super::InstanceRef;

    #[test]
    fn serde_view_preserves_json_semantic_equality() {
        let value = json!({"name": "Ada", "values": [1, 2.0, true, null]});
        assert!(InstanceRef::from_serde(&value).equals(&value));
    }

    #[cfg(feature = "jiter")]
    #[test]
    fn jiter_view_preserves_json_semantic_equality() {
        let parsed =
            jiter::JsonValue::parse(br#"{"name":"Ada","values":[1,2.0,true,null]}"#, false)
                .unwrap();
        let expected = json!({"name": "Ada", "values": [1, 2, true, null]});
        assert!(InstanceRef::from_jiter(&parsed).equals(&expected));
    }

    #[cfg(feature = "jiter")]
    #[test]
    fn jiter_flag_validation_matches_serde_validation() {
        let schema = json!({
            "$defs": {
                "node": {
                    "oneOf": [
                        {
                            "type": "object",
                            "required": ["kind", "name"],
                            "properties": {
                                "kind": {"const": "leaf"},
                                "name": {"type": "string", "minLength": 1}
                            },
                            "additionalProperties": false
                        },
                        {
                            "type": "object",
                            "required": ["kind", "children"],
                            "properties": {
                                "kind": {"const": "branch"},
                                "children": {
                                    "type": "array",
                                    "items": {"$ref": "#/$defs/node"}
                                }
                            },
                            "additionalProperties": false
                        }
                    ]
                }
            },
            "$ref": "#/$defs/node"
        });
        let validator = crate::draft202012::options().build(&schema).unwrap();
        for source in [
            r#"{"kind":"leaf","name":"Ada"}"#,
            r#"{"kind":"branch","children":[{"kind":"leaf","name":"Ada"}]}"#,
            r#"{"kind":"leaf","name":""}"#,
            r#"{"kind":"branch","children":[{"kind":"unknown"}]}"#,
            r#"{"kind":"leaf","name":"Ada","extra":true}"#,
        ] {
            let serde: Value = serde_json::from_str(source).unwrap();
            let jiter = jiter::JsonValue::parse(source.as_bytes(), false).unwrap();
            assert_eq!(
                validator.is_valid(&serde),
                validator.is_valid_instance(InstanceRef::from_jiter(&jiter)),
                "representations disagreed for {source}"
            );
        }
    }

    #[cfg(feature = "jiter")]
    #[test]
    fn jiter_numeric_validation_matches_serde_at_large_fractional_boundary() {
        for (schema_source, instance_source, expected) in [
            (
                r#"{"minimum":9.727837981879871e+26}"#,
                r"9.727837981879871e+26",
                true,
            ),
            (
                r#"{"maximum":9.727837981879871e+26}"#,
                r"9.727837981879871e+26",
                true,
            ),
            (
                r#"{"exclusiveMinimum":9.727837981879871e+26}"#,
                r"9.727837981879871e+26",
                false,
            ),
            (
                r#"{"exclusiveMaximum":9.727837981879871e+26}"#,
                r"9.727837981879871e+26",
                false,
            ),
            (r#"{"multipleOf":0.1}"#, "0.3", true),
        ] {
            let schema: Value = serde_json::from_str(schema_source).unwrap();
            let serde: Value = serde_json::from_str(instance_source).unwrap();
            let jiter = jiter::JsonValue::parse(instance_source.as_bytes(), false).unwrap();
            let validator = crate::draft202012::options().build(&schema).unwrap();

            assert_eq!(validator.is_valid(&serde), expected, "{schema_source}");
            assert_eq!(
                validator.is_valid(&serde),
                validator.is_valid_instance(InstanceRef::from_jiter(&jiter)),
                "{schema_source}"
            );
        }
    }

    #[cfg(feature = "python")]
    #[test]
    fn python_numeric_validation_matches_serde_at_large_fractional_boundary() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let json = py.import("json").unwrap();
            for (schema_source, instance_source, expected) in [
                (
                    r#"{"minimum":9.727837981879871e+26}"#,
                    r"9.727837981879871e+26",
                    true,
                ),
                (
                    r#"{"maximum":9.727837981879871e+26}"#,
                    r"9.727837981879871e+26",
                    true,
                ),
                (
                    r#"{"exclusiveMinimum":9.727837981879871e+26}"#,
                    r"9.727837981879871e+26",
                    false,
                ),
                (
                    r#"{"exclusiveMaximum":9.727837981879871e+26}"#,
                    r"9.727837981879871e+26",
                    false,
                ),
                (r#"{"multipleOf":0.1}"#, "0.3", true),
            ] {
                let schema: Value = serde_json::from_str(schema_source).unwrap();
                let serde: Value = serde_json::from_str(instance_source).unwrap();
                let python = json.call_method1("loads", (instance_source,)).unwrap();
                let validator = crate::draft202012::options().build(&schema).unwrap();

                assert_eq!(validator.is_valid(&serde), expected, "{schema_source}");
                assert_eq!(
                    validator.is_valid(&serde),
                    validator.is_valid_instance(InstanceRef::from_python(&python)),
                    "{schema_source}"
                );
            }
        });
    }

    #[cfg(feature = "jiter")]
    #[test]
    fn allocation_backed_fallback_does_not_reuse_temporary_instance_cache() {
        let schema = json!({
            "type": "array",
            "items": {
                "contains": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"]
                }
            }
        });
        let validator = crate::draft202012::options().build(&schema).unwrap();
        let source = r#"[[{"value":1}],[{"value":"invalid"}]]"#;
        let serde: Value = serde_json::from_str(source).unwrap();
        let jiter = jiter::JsonValue::parse(source.as_bytes(), false).unwrap();

        assert!(!validator.is_valid(&serde));
        assert_eq!(
            validator.is_valid(&serde),
            validator.is_valid_instance(InstanceRef::from_jiter(&jiter))
        );
    }
}
