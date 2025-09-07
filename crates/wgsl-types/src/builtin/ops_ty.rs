//! Type-checking of operators.

use crate::{
    Error,
    conv::{Convert, convert_ty},
    ty::{Ty, Type},
};

use wgsl_syntax::{BinaryOperator, UnaryOperator};

type E = Error;

// -------------------
// LOGICAL EXPRESSIONS
// -------------------
// reference: https://www.w3.org/TR/WGSL/#logical-expr

impl Type {
    pub fn op_not(&self) -> Result<Self, E> {
        match self {
            Self::Bool | Self::Vec(_, _) => Ok(self.clone()),
            _ => Err(E::Unary(UnaryOperator::LogicalNegation, self.clone())),
        }
    }
}

// ----------------------
// ARITHMETIC EXPRESSIONS
// ----------------------
// reference: https://www.w3.org/TR/WGSL/#arithmetic-expr

impl Type {
    /// Valid operands:
    /// * `-S`, S: scalar
    pub fn op_neg(&self) -> Result<Self, E> {
        if self.is_scalar() {
            Ok(self.clone())
        } else {
            Err(E::Unary(UnaryOperator::Negation, self.ty()))
        }
    }

    /// Valid operands:
    /// * `T + T`, T: scalar or vec
    /// * `S + V` or `V + S`, S: scalar, V: vec<S>
    /// * `M + M`, M: mat
    pub fn op_add(&self, rhs: &Type) -> Result<Self, E> {
        let err = || E::Binary(BinaryOperator::Addition, self.ty(), rhs.ty());
        match (self, rhs) {
            (lhs, rhs) if lhs.is_scalar() && rhs.is_scalar() || lhs.is_vec() && rhs.is_vec() => {
                let ty = convert_ty(self, rhs).ok_or_else(err)?;
                Ok(ty.clone())
            }
            (scalar_ty, Type::Vec(n, vec_ty)) | (Type::Vec(n, vec_ty), scalar_ty)
                if scalar_ty.is_scalar() =>
            {
                let inner_ty = convert_ty(scalar_ty, vec_ty).ok_or_else(err)?;
                Ok(Type::Vec(*n, Box::new(inner_ty.clone())))
            }
            (Type::Mat(c1, r1, lhs), Type::Mat(c2, r2, rhs)) if c1 == c2 && r1 == r2 => {
                let inner_ty = convert_ty(lhs, rhs).ok_or_else(err)?;
                Ok(Type::Mat(*c1, *c2, Box::new(inner_ty.clone())))
            }
            _ => Err(err()),
        }
    }

    /// Valid operands:
    /// * `T - T`, T: scalar or vec
    /// * `S - V` or `V - S`, S: scalar, V: `vec<S>`
    /// * `M - M`, M: mat
    pub fn op_sub(&self, rhs: &Type) -> Result<Self, E> {
        let err = || E::Binary(BinaryOperator::Subtraction, self.ty(), rhs.ty());
        match (self, rhs) {
            (lhs, rhs) if lhs.is_scalar() && rhs.is_scalar() || lhs.is_vec() && rhs.is_vec() => {
                let ty = convert_ty(self, rhs).ok_or_else(err)?;
                Ok(ty.clone())
            }
            (scalar_ty, Type::Vec(n, vec_ty)) | (Type::Vec(n, vec_ty), scalar_ty)
                if scalar_ty.is_scalar() =>
            {
                let inner_ty = convert_ty(scalar_ty, vec_ty).ok_or_else(err)?;
                Ok(Type::Vec(*n, Box::new(inner_ty.clone())))
            }
            (Type::Mat(c1, r1, lhs), Type::Mat(c2, r2, rhs)) if c1 == c2 && r1 == r2 => {
                let inner_ty = convert_ty(lhs, rhs).ok_or_else(err)?;
                Ok(Type::Mat(*c1, *c2, Box::new(inner_ty.clone())))
            }
            _ => Err(err()),
        }
    }

    /// Valid operands:
    /// * `T * T`, T: scalar or vec
    /// * `S * V` or `V * S`, S: scalar, V: `vec<S>`
    /// * `S * M` or `M * S`, S: float, M: `mat<S>`
    /// * `V * M` or `M * V`, S: float, V: `vec<S>`, M: `mat<S>`
    /// * `M1 * M1`, M1: `matKxR`, M2: `matCxK`
    pub fn op_mul(&self, rhs: &Type) -> Result<Self, E> {
        let err = || E::Binary(BinaryOperator::Multiplication, self.ty(), rhs.ty());
        match (self, rhs) {
            (lhs, rhs) if lhs.is_scalar() && rhs.is_scalar() || lhs.is_vec() && rhs.is_vec() => {
                let ty = convert_ty(self, rhs).ok_or_else(err)?;
                Ok(ty.clone())
            }
            (scalar_ty, Type::Vec(n, vec_ty)) | (Type::Vec(n, vec_ty), scalar_ty)
                if scalar_ty.is_scalar() =>
            {
                let inner_ty = convert_ty(scalar_ty, vec_ty).ok_or_else(err)?;
                Ok(Type::Vec(*n, Box::new(inner_ty.clone())))
            }
            (scalar_ty, Type::Mat(c, r, mat_ty)) | (Type::Mat(c, r, mat_ty), scalar_ty)
                if scalar_ty.is_scalar() =>
            {
                let inner_ty = convert_ty(scalar_ty, mat_ty).ok_or_else(err)?;
                Ok(Type::Mat(*c, *r, Box::new(inner_ty.clone())))
            }
            (Type::Vec(n1, vec_ty), Type::Mat(n2, n, mat_ty))
            | (Type::Mat(n, n1, mat_ty), Type::Vec(n2, vec_ty))
                if n1 == n2 =>
            {
                let inner_ty = convert_ty(vec_ty, mat_ty).ok_or_else(err)?;
                Ok(Type::Vec(*n, Box::new(inner_ty.clone())))
            }
            (Type::Mat(k1, r, lhs), Type::Mat(c, k2, rhs)) if k1 == k2 => {
                let inner_ty = convert_ty(lhs, rhs).ok_or_else(err)?;
                Ok(Type::Mat(*c, *r, Box::new(inner_ty.clone())))
            }
            _ => Err(err()),
        }
    }

    /// Valid operands:
    /// * `T / T`, T: scalar or vec
    /// * `S / V` or `V / S`, S: scalar, V: `vec<S>`
    pub fn op_div(&self, rhs: &Type) -> Result<Self, E> {
        let err = || E::Binary(BinaryOperator::Division, self.ty(), rhs.ty());
        match (self, rhs) {
            (lhs, rhs) if lhs.is_scalar() && rhs.is_scalar() || lhs.is_vec() && rhs.is_vec() => {
                let ty = convert_ty(self, rhs).ok_or_else(err)?;
                Ok(ty.clone())
            }
            (scalar_ty, Type::Vec(n, vec_ty)) | (Type::Vec(n, vec_ty), scalar_ty)
                if scalar_ty.is_scalar() =>
            {
                let inner_ty = convert_ty(scalar_ty, vec_ty).ok_or_else(err)?;
                Ok(Type::Vec(*n, Box::new(inner_ty.clone())))
            }
            _ => Err(err()),
        }
    }

    /// Valid operands:
    /// * `T % T`, T: scalar or vec
    /// * `S % V` or `V % S`, S: scalar, V: `vec<S>`
    pub fn op_rem(&self, rhs: &Type) -> Result<Self, E> {
        let err = || E::Binary(BinaryOperator::Remainder, self.ty(), rhs.ty());
        match (self, rhs) {
            (lhs, rhs) if lhs.is_scalar() && rhs.is_scalar() || lhs.is_vec() && rhs.is_vec() => {
                let ty = convert_ty(self, rhs).ok_or_else(err)?;
                Ok(ty.clone())
            }
            (scalar_ty, Type::Vec(n, vec_ty)) | (Type::Vec(n, vec_ty), scalar_ty)
                if scalar_ty.is_scalar() =>
            {
                let inner_ty = convert_ty(scalar_ty, vec_ty).ok_or_else(err)?;
                Ok(Type::Vec(*n, Box::new(inner_ty.clone())))
            }
            _ => Err(err()),
        }
    }
}

// ----------------------
// COMPARISON EXPRESSIONS
// ----------------------
// reference: https://www.w3.org/TR/WGSL/#comparison-expr

impl Type {
    /// Valid operands:
    /// * `T == T`, T: scalar or vec
    pub fn op_eq(&self, rhs: &Type) -> Result<Type, E> {
        let err = || E::Binary(BinaryOperator::Equality, self.ty(), rhs.ty());
        match convert_ty(self, rhs).ok_or_else(err)? {
            ty if ty.is_scalar() => Ok(Type::Bool),
            Type::Vec(n, _) => Ok(Type::Vec(*n, Box::new(Type::Bool))),
            _ => Err(err()),
        }
    }
    /// Valid operands:
    /// * `T != T`, T: scalar or vec
    pub fn op_ne(&self, rhs: &Type) -> Result<Type, E> {
        let err = || E::Binary(BinaryOperator::Inequality, self.ty(), rhs.ty());
        match convert_ty(self, rhs).ok_or_else(err)? {
            ty if ty.is_scalar() => Ok(Type::Bool),
            Type::Vec(n, _) => Ok(Type::Vec(*n, Box::new(Type::Bool))),
            _ => Err(err()),
        }
    }
    /// Valid operands:
    /// * `T < T`, T: scalar or vec
    pub fn op_lt(&self, rhs: &Type) -> Result<Type, E> {
        let err = || E::Binary(BinaryOperator::LessThan, self.ty(), rhs.ty());
        match convert_ty(self, rhs).ok_or_else(err)? {
            ty if ty.is_scalar() => Ok(Type::Bool),
            Type::Vec(n, _) => Ok(Type::Vec(*n, Box::new(Type::Bool))),
            _ => Err(err()),
        }
    }
    /// Valid operands:
    /// * `T <= T`, T: scalar or vec
    pub fn op_le(&self, rhs: &Type) -> Result<Type, E> {
        let err = || E::Binary(BinaryOperator::LessThanEqual, self.ty(), rhs.ty());
        match convert_ty(self, rhs).ok_or_else(err)? {
            ty if ty.is_scalar() => Ok(Type::Bool),
            Type::Vec(n, _) => Ok(Type::Vec(*n, Box::new(Type::Bool))),
            _ => Err(err()),
        }
    }
    /// Valid operands:
    /// * `T > T`, T: scalar or vec
    pub fn op_gt(&self, rhs: &Type) -> Result<Type, E> {
        let err = || E::Binary(BinaryOperator::GreaterThan, self.ty(), rhs.ty());
        match convert_ty(self, rhs).ok_or_else(err)? {
            ty if ty.is_scalar() => Ok(Type::Bool),
            Type::Vec(n, _) => Ok(Type::Vec(*n, Box::new(Type::Bool))),
            _ => Err(err()),
        }
    }
    /// Valid operands:
    /// * `T >= T`, T: scalar or vec
    pub fn op_ge(&self, rhs: &Type) -> Result<Type, E> {
        let err = || E::Binary(BinaryOperator::GreaterThanEqual, self.ty(), rhs.ty());
        match convert_ty(self, rhs).ok_or_else(err)? {
            ty if ty.is_scalar() => Ok(Type::Bool),
            Type::Vec(n, _) => Ok(Type::Vec(*n, Box::new(Type::Bool))),
            _ => Err(err()),
        }
    }
}

// ---------------
// BIT EXPRESSIONS
// ---------------
// reference: https://www.w3.org/TR/WGSL/#bit-expr

impl Type {
    /// Valid operands:
    /// * `~T`, I: integer, T: I or `vec<I>`
    pub fn op_bitnot(&self) -> Result<Type, E> {
        match self {
            ty if ty.is_integer() => Ok(self.clone()),
            Type::Vec(_, ty) if ty.is_integer() => Ok(self.clone()),
            _ => Err(E::Unary(UnaryOperator::BitwiseComplement, self.ty())),
        }
    }

    /// Valid operands:
    /// * `T | T`, I: integer, T: I or `vec<I>`
    pub fn op_bitor(&self, rhs: &Type) -> Result<Type, E> {
        let err = || E::Binary(BinaryOperator::BitwiseOr, self.ty(), rhs.ty());
        match convert_ty(self, rhs).ok_or_else(err)? {
            ty if ty.is_integer() => Ok(self.clone()),
            Type::Vec(_, ty) if ty.is_integer() => Ok(self.clone()),
            _ => Err(err()),
        }
    }

    /// Valid operands:
    /// * `T & T`, I: integer, T: I or `vec<I>`
    pub fn op_bitand(&self, rhs: &Type) -> Result<Type, E> {
        let err = || E::Binary(BinaryOperator::BitwiseAnd, self.ty(), rhs.ty());
        match convert_ty(self, rhs).ok_or_else(err)? {
            ty if ty.is_integer() => Ok(self.clone()),
            Type::Vec(_, ty) if ty.is_integer() => Ok(self.clone()),
            _ => Err(err()),
        }
    }

    /// Valid operands:
    /// * `T ^ T`, T: integer or `vec<integer>`
    pub fn op_bitxor(&self, rhs: &Type) -> Result<Type, E> {
        let err = || E::Binary(BinaryOperator::BitwiseXor, self.ty(), rhs.ty());
        match convert_ty(self, rhs).ok_or_else(err)? {
            ty if ty.is_integer() => Ok(self.clone()),
            Type::Vec(_, ty) if ty.is_integer() => Ok(self.clone()),
            _ => Err(err()),
        }
    }

    /// Valid operands:
    /// * `integer << u32`
    /// * `vec<integer> << vec<u32>`
    pub fn op_shl(&self, rhs: &Type) -> Result<Type, E> {
        let err = || E::Binary(BinaryOperator::ShiftLeft, self.ty(), rhs.ty());
        let rhs = rhs.convert_inner_to(&Type::U32).ok_or_else(err)?;
        match (self, rhs) {
            (lhs, Type::U32) if lhs.is_integer() => Ok(lhs.clone()),
            (lhs, Type::Vec(_, _)) => Ok(lhs.clone()),
            _ => Err(err()),
        }
    }

    /// Valid operands:
    /// * `integer >> u32`
    /// * `vec<integer> >> vec<u32>`
    pub fn op_shr(&self, rhs: &Type) -> Result<Type, E> {
        let err = || E::Binary(BinaryOperator::ShiftRight, self.ty(), rhs.ty());
        let rhs = rhs.convert_inner_to(&Type::U32).ok_or_else(err)?;
        match (self, rhs) {
            (lhs, Type::U32) if lhs.is_integer() => Ok(lhs.clone()),
            (lhs, Type::Vec(_, _)) => Ok(lhs.clone()),
            _ => Err(err()),
        }
    }
}
