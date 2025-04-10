use std::{borrow::Cow, collections::HashMap, iter::Iterator};

use crate::{builtin::builtin_ident, visit::Visit};
use wesl_macros::query_mut;
use wgsl_parse::syntax::*;

/// was that not in the std at some point???
type BoxedIterator<'a, T> = Box<dyn Iterator<Item = T> + 'a>;

#[allow(dead_code)]
pub trait IteratorExt: Iterator {
    fn boxed<'a>(self) -> BoxedIterator<'a, Self::Item>
    where
        Self: Sized + 'a;
}

impl<T: Iterator> IteratorExt for T {
    fn boxed<'a>(self) -> BoxedIterator<'a, Self::Item>
    where
        Self: Sized + 'a,
    {
        Box::new(self)
    }
}

pub trait SyntaxUtil {
    fn entry_points(&self) -> impl Iterator<Item = &Ident>;
    fn retarget_idents(&mut self);
}

impl SyntaxUtil for TranslationUnit {
    fn entry_points(&self) -> impl Iterator<Item = &Ident> {
        self.global_declarations
            .iter()
            .filter_map(|decl| match decl.node() {
                GlobalDeclaration::Function(decl) => decl
                    .attributes
                    .iter()
                    .any(|attr| {
                        matches!(
                            attr,
                            Attribute::Vertex | Attribute::Fragment | Attribute::Compute
                        )
                    })
                    .then_some(&decl.ident),
                _ => None,
            })
    }

    /// make all identifiers that point to the same declaration refer to the same string.
    ///
    /// retarget local references to the local declaration ident and global
    /// references to the global declaration ident. It does this by keeping track of the
    /// local declarations scope.
    fn retarget_idents(&mut self) {
        // keep track of declarations in a scope.
        type Scope<'a> = Cow<'a, HashMap<String, Ident>>;

        fn flatten_imports(imports: &[ImportStatement]) -> impl Iterator<Item = Ident> + '_ {
            fn rec(content: &ImportContent) -> impl Iterator<Item = Ident> + '_ {
                match &content {
                    ImportContent::Item(item) => {
                        std::iter::once(item.rename.as_ref().unwrap_or(&item.ident).clone()).boxed()
                    }
                    ImportContent::Collection(coll) => {
                        coll.iter().flat_map(|import| rec(&import.content)).boxed()
                    }
                }
            }
            imports.iter().flat_map(|import| rec(&import.content))
        }

        let scope: Scope = Cow::Owned(
            self.global_declarations
                .iter()
                .filter_map(|decl| decl.ident())
                .map(|id| (id.to_string(), id.clone()))
                .chain(flatten_imports(&self.imports).map(|id| (id.to_string(), id)))
                .collect::<HashMap<_, _>>(),
        );

        fn retarget_ty(ty: &mut TypeExpression, scope: &Scope) {
            if let Some((_, id)) = scope
                .iter()
                .find(|(name, _)| name.as_str() == *ty.ident.name())
            {
                ty.ident = id.clone();
            } else {
                let builtin = builtin_ident(&ty.ident.name()).cloned();
                if let Some(id) = builtin {
                    ty.ident = id;
                }
            }
            query_mut!(ty.template_args.[].[].expression.(x => Visit::<TypeExpression>::visit_mut(&mut **x)))
                .for_each(|ty| retarget_ty(ty, scope));
        }

        // retarget local references to the local declaration ident and global
        // references to the global declaration ident. It does this by keeping track of the
        // local declarations scope.
        fn retarget_stats<'a>(
            stats: impl IntoIterator<Item = &'a mut StatementNode>,
            mut scope: Scope<'a>,
        ) -> Scope<'a> {
            stats.into_iter().for_each(|stmt| match stmt.node_mut() {
                Statement::Void => (),
                Statement::Compound(s) => {
                    query_mut!(s.attributes.[].(x => x.visit_mut()))
                        .for_each(|ty| retarget_ty(ty, &scope));
                    retarget_stats(&mut s.statements, scope.clone());
                }
                Statement::Assignment(s) => {
                    query_mut!(s.{
                        attributes.[].(x => x.visit_mut()),
                        lhs.(x => Visit::<TypeExpression>::visit_mut(&mut **x)),
                        rhs.(x => Visit::<TypeExpression>::visit_mut(&mut **x)),
                    })
                    .for_each(|ty| retarget_ty(ty, &scope));
                }
                Statement::Increment(s) => {
                    query_mut!(s.{
                        attributes.[].(x => x.visit_mut()),
                        expression.(x => Visit::<TypeExpression>::visit_mut(&mut **x)),
                    })
                    .for_each(|ty| retarget_ty(ty, &scope));
                }
                Statement::Decrement(s) => {
                    query_mut!(s.{
                        attributes.[].(x => x.visit_mut()),
                        expression.(x => Visit::<TypeExpression>::visit_mut(&mut **x)),
                    })
                    .for_each(|ty| retarget_ty(ty, &scope));
                }
                Statement::If(s) => {
                    let s2 = &mut *s; // COMBAK: not sure why this is needed?
                    query_mut!(s2.{
                        attributes.[].(x => x.visit_mut()),
                        if_clause.{
                            expression.(x => Visit::<TypeExpression>::visit_mut(&mut **x)),
                            body.{
                                attributes.[].(x => x.visit_mut()),
                            }
                        },
                        else_if_clauses.[].{
                            attributes.[].(x => x.visit_mut()),
                            expression.(x => Visit::<TypeExpression>::visit_mut(&mut **x)),
                            body.{
                                attributes.[].(x => x.visit_mut()),
                            }
                        },
                        else_clause.[].{
                            attributes.[].(x => x.visit_mut()),
                            body.{
                                attributes.[].(x => x.visit_mut()),
                            },
                        },
                    })
                    .for_each(|ty| retarget_ty(ty, &scope));
                    retarget_stats(&mut s.if_clause.body.statements, scope.clone());
                    for clause in &mut s.else_if_clauses {
                        retarget_stats(&mut clause.body.statements, scope.clone());
                    }
                    if let Some(clause) = &mut s.else_clause {
                        retarget_stats(&mut clause.body.statements, scope.clone());
                    }
                }
                Statement::Switch(s) => {
                    let s2 = &mut *s; // COMBAK: not sure why this is needed?
                    query_mut!(s2.{
                        attributes.[].(x => x.visit_mut()),
                        expression.(x => Visit::<TypeExpression>::visit_mut(&mut **x)),
                        body_attributes.[].(x => x.visit_mut()),
                        clauses.[].{
                            attributes.[].(x => x.visit_mut()),
                            case_selectors.[].CaseSelector::Expression.(x => Visit::<TypeExpression>::visit_mut(&mut **x)),
                            body.{
                                attributes.[].(x => x.visit_mut()),
                            }
                        },

                    })
                    .for_each(|ty| retarget_ty(ty, &scope));
                    for clause in &mut s.clauses {
                        retarget_stats(&mut clause.body.statements, scope.clone());
                    }
                }
                Statement::Loop(s) => {
                    let s2 = &mut *s; // COMBAK: not sure why this is needed?
                    query_mut!(s2.{
                        attributes.[].(x => x.visit_mut()),
                        body.attributes.[].(x => x.visit_mut()),
                    })
                    .for_each(|ty| retarget_ty(ty, &scope));
                    let scope = retarget_stats(&mut s.body.statements, scope.clone());
                    // continuing, if present, must be the last statement of the loop body
                    // and therefore has access to the scope at the end of the body.
                    if let Some(s) = &mut s.continuing {
                        let s2 = &mut *s; // COMBAK: not sure why this is needed?
                        query_mut!(s2.{
                            attributes.[].(x => x.visit_mut()),
                            body.attributes.[].(x => x.visit_mut()),
                        })
                        .for_each(|ty| retarget_ty(ty, &scope));
                        let scope = retarget_stats(&mut s.body.statements, scope.clone());
                        // break-if, if present, must be the last statement of the continuing body
                        // and therefore has access to the scope at the end of the body.
                        if let Some(s) = &mut s.break_if {
                            let s2 = &mut *s; // COMBAK: not sure why this is needed?
                            query_mut!(s2.{
                                attributes.[].(x => x.visit_mut()),
                                expression.(x => Visit::<TypeExpression>::visit_mut(&mut **x)),
                            })
                            .for_each(|ty| retarget_ty(ty, &scope));
                        }
                    }
                }
                Statement::For(s) => {
                    query_mut!(s.attributes.[].(x => x.visit_mut()))
                        .for_each(|ty| retarget_ty(ty, &scope));
                    let scope = if let Some(init) = &mut s.initializer {
                        retarget_stats([init], scope.clone())
                    } else {
                        scope.clone()
                    };
                    query_mut!(s.condition.[].(x => Visit::<TypeExpression>::visit_mut(&mut **x)))
                        .for_each(|ty| retarget_ty(ty, &scope));
                    query_mut!(s.body.attributes.[].(x => x.visit_mut()))
                        .for_each(|ty| retarget_ty(ty, &scope));
                    if let Some(update) = &mut s.update {
                        retarget_stats([update], scope.clone());
                    }
                    retarget_stats(&mut s.body.statements, scope);
                }
                Statement::While(s) => {
                    let s2 = &mut *s; // COMBAK: not sure why this is needed?
                    query_mut!(s2.{
                        attributes.[].(x => x.visit_mut()),
                        condition.(x => Visit::<TypeExpression>::visit_mut(&mut **x)),
                        body.attributes.[].(x => x.visit_mut()),
                    })
                    .for_each(|ty| retarget_ty(ty, &scope));
                    retarget_stats(&mut s.body.statements, scope.clone());
                }
                Statement::Break(s) => {
                    query_mut!(s.attributes.[].(x => x.visit_mut()))
                        .for_each(|ty| retarget_ty(ty, &scope));
                }
                Statement::Continue(s) => {
                    query_mut!(s.attributes.[].(x => x.visit_mut()))
                        .for_each(|ty| retarget_ty(ty, &scope));
                }
                Statement::Return(s) => {
                    query_mut!(s.expression.[].(x => Visit::<TypeExpression>::visit_mut(&mut **x)))
                        .for_each(|ty| retarget_ty(ty, &scope));
                }
                Statement::Discard(s) => {
                    query_mut!(s.attributes.[].(x => x.visit_mut()))
                        .for_each(|ty| retarget_ty(ty, &scope));
                }
                Statement::FunctionCall(s) => {
                    query_mut!(s.{
                        attributes.[].(x => x.visit_mut()),
                        call.{
                            ty,
                            arguments.[].(x => Visit::<TypeExpression>::visit_mut(&mut **x)),
                        }
                    })
                    .for_each(|ty| retarget_ty(ty, &scope));
                }
                Statement::ConstAssert(s) => {
                    query_mut!(s.{
                        expression.(x => Visit::<TypeExpression>::visit_mut(&mut **x))
                    })
                    .for_each(|ty| retarget_ty(ty, &scope));
                }
                Statement::Declaration(s) => {
                    let s2 = &mut *s; // COMBAK: not sure why this is needed?
                    query_mut!(s2.{
                        attributes.[].(x => x.visit_mut()),
                        ty.[],
                        initializer.[].(x => Visit::<TypeExpression>::visit_mut(&mut **x)),
                    })
                    .for_each(|ty| retarget_ty(ty, &scope));
                    scope.to_mut().insert(s.ident.to_string(), s.ident.clone());
                }
            });
            scope
        }

        for decl in &mut self.global_declarations {
            match decl.node_mut() {
                GlobalDeclaration::Void => (),
                GlobalDeclaration::Declaration(d) => {
                    Visit::<TypeExpression>::visit_mut(d).for_each(|ty| retarget_ty(ty, &scope))
                }
                GlobalDeclaration::TypeAlias(d) => {
                    Visit::<TypeExpression>::visit_mut(d).for_each(|ty| retarget_ty(ty, &scope))
                }
                GlobalDeclaration::Struct(d) => {
                    Visit::<TypeExpression>::visit_mut(d).for_each(|ty| retarget_ty(ty, &scope))
                }
                GlobalDeclaration::Function(d) => {
                    #[cfg(feature = "generics")]
                    let scope = {
                        let mut scope = scope.clone();
                        scope
                            .to_mut()
                            .extend(d.attributes.iter().filter_map(|attr| match attr {
                                Attribute::Type(attr) => {
                                    Some((attr.ident.to_string(), attr.ident.clone()))
                                }
                                _ => None,
                            }));
                        scope
                    };
                    let d2 = &mut *d; // COMBAK: not sure why this is needed?
                    query_mut!(d2.{
                        attributes.[].(x => x.visit_mut()),
                        parameters.[].{
                            attributes.[].(x => x.visit_mut()),
                            ty,
                        },
                        return_attributes.[].(x => x.visit_mut()),
                        return_type.[],
                        body.{
                            attributes.[].(x => x.visit_mut()),
                        }
                    })
                    .for_each(|ty| retarget_ty(ty, &scope));
                    let mut scope = scope.clone();
                    scope.to_mut().extend(
                        d.parameters
                            .iter()
                            .map(|param| (param.ident.to_string(), param.ident.clone())),
                    );
                    retarget_stats(&mut d.body.statements, scope);
                }
                GlobalDeclaration::ConstAssert(d) => {
                    Visit::<TypeExpression>::visit_mut(d).for_each(|ty| retarget_ty(ty, &scope))
                }
            }
        }
    }
}
