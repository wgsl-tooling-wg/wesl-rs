use wgsl_parse::{
    SyntaxNode,
    syntax::{Attribute, AttributeNode, BuiltinValue, Expression},
};
use wgsl_types::{
    ShaderStage,
    inst::{Instance, LiteralInstance},
    ty::{Ty, Type},
};

use super::{Context, Eval, EvalError, with_stage};

type E = EvalError;

pub trait EvalAttrs: SyntaxNode {
    fn attr_align(&self, ctx: &mut Context) -> Result<Option<u32>, E> {
        attr_align(self.attributes(), ctx).transpose()
    }
    fn attr_group_binding(&self, ctx: &mut Context) -> Result<(u32, u32), E> {
        attr_group_binding(self.attributes(), ctx)
    }
    fn attr_size(&self, ctx: &mut Context) -> Result<Option<u32>, E> {
        attr_size(self.attributes(), ctx).transpose()
    }
    fn attr_id(&self, ctx: &mut Context) -> Result<Option<u32>, E> {
        attr_id(self.attributes(), ctx).transpose()
    }
    fn attr_location(&self, ctx: &mut Context) -> Result<Option<u32>, E> {
        attr_location(self.attributes(), ctx).transpose()
    }
    fn attr_workgroup_size(&self, ctx: &mut Context) -> Result<(u32, Option<u32>, Option<u32>), E> {
        attr_workgroup_size(self.attributes(), ctx)
    }
    fn attr_blend_src(&self, ctx: &mut Context) -> Result<Option<bool>, E> {
        attr_blend_src(self.attributes(), ctx).transpose()
    }
    fn attr_builtin(&self) -> Option<BuiltinValue> {
        self.attributes().iter().find_map(|attr| match attr.node() {
            Attribute::Builtin(attr) => Some(*attr),
            _ => None,
        })
    }
}

impl<T: SyntaxNode> EvalAttrs for T {}
fn eval_positive_integer(expr: &Expression, ctx: &mut Context) -> Result<u32, E> {
    let inst = with_stage!(ctx, ShaderStage::Const, { expr.eval_value(ctx) })?;
    let integer = match inst {
        Instance::Literal(g) => match g {
            LiteralInstance::AbstractInt(g) => Ok(g),
            LiteralInstance::I32(g) => Ok(g as i64),
            LiteralInstance::U32(g) => Ok(g as i64),
            _ => Err(E::Type(Type::U32, g.ty())),
        },
        _ => Err(E::Type(Type::U32, inst.ty())),
    }?;
    if integer < 0 {
        Err(E::NegativeAttr(integer))
    } else {
        Ok(integer as u32)
    }
}

fn attr_group_binding(attrs: &[AttributeNode], ctx: &mut Context) -> Result<(u32, u32), E> {
    let group = attrs.iter().find_map(|attr| match attr.node() {
        Attribute::Group(g) => Some(g),
        _ => None,
    });
    let binding = attrs.iter().find_map(|attr| match attr.node() {
        Attribute::Binding(b) => Some(b),
        _ => None,
    });

    let (group, binding) = match (group, binding) {
        (Some(g), Some(b)) => Ok((
            eval_positive_integer(g, ctx)?,
            eval_positive_integer(b, ctx)?,
        )),
        _ => Err(E::MissingBindAttr),
    }?;
    Ok((group, binding))
}

fn attr_size(attrs: &[AttributeNode], ctx: &mut Context) -> Option<Result<u32, E>> {
    let expr = attrs.iter().find_map(|attr| match attr.node() {
        Attribute::Size(e) => Some(e),
        _ => None,
    })?;

    Some(eval_positive_integer(expr, ctx))
}

fn attr_align(attrs: &[AttributeNode], ctx: &mut Context) -> Option<Result<u32, E>> {
    let expr = attrs.iter().find_map(|attr| match attr.node() {
        Attribute::Align(e) => Some(e),
        _ => None,
    })?;

    Some(eval_positive_integer(expr, ctx))
}

fn attr_id(attrs: &[AttributeNode], ctx: &mut Context) -> Option<Result<u32, E>> {
    let expr = attrs.iter().find_map(|attr| match attr.node() {
        Attribute::Id(e) => Some(e),
        _ => None,
    })?;

    Some(eval_positive_integer(expr, ctx))
}

fn attr_location(attrs: &[AttributeNode], ctx: &mut Context) -> Option<Result<u32, E>> {
    let expr = attrs.iter().find_map(|attr| match attr.node() {
        Attribute::Location(e) => Some(e),
        _ => None,
    })?;

    Some(eval_positive_integer(expr, ctx))
}

fn attr_workgroup_size(
    attrs: &[AttributeNode],
    ctx: &mut Context,
) -> Result<(u32, Option<u32>, Option<u32>), E> {
    let attr = attrs
        .iter()
        .find_map(|attr| match attr.node() {
            Attribute::WorkgroupSize(attr) => Some(attr),
            _ => None,
        })
        .ok_or(E::MissingWorkgroupSize)?;

    let x = eval_positive_integer(&attr.x, ctx)?;
    let y = attr
        .y
        .as_ref()
        .map(|y| eval_positive_integer(y, ctx))
        .transpose()?;
    let z = attr
        .z
        .as_ref()
        .map(|z| eval_positive_integer(z, ctx))
        .transpose()?;
    Ok((x, y, z))
}

fn attr_blend_src(attrs: &[AttributeNode], ctx: &mut Context) -> Option<Result<bool, E>> {
    let expr = attrs.iter().find_map(|attr| match attr.node() {
        Attribute::BlendSrc(attr) => Some(attr),
        _ => None,
    })?;
    Some(eval_positive_integer(expr, ctx).and_then(|val| match val {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(E::InvalidBlendSrc(val)),
    }))
}
