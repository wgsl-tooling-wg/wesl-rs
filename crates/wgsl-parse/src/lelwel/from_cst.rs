use crate::{
    Error,
    lelwel::parser::{Cst, Node, NodeRef, Rule},
    syntax::*,
};

pub(crate) trait FromCst: Sized {
    fn from_cst(cst: &Cst, node: NodeRef) -> Self;
}

impl FromCst for TranslationUnit {
    fn from_cst(cst: &Cst, node: NodeRef) -> Self {
        if !cst.match_rule(node, Rule::TranslationUnit) {
            panic!("expected a TranslationUnit, got")
        };

        let mut imports = Vec::new();
        let mut global_directives = Vec::new();
        let mut global_declarations = Vec::new();

        for child in cst.children(node) {
            if false {
                // TODO imports
            } else if cst.match_rule(child, Rule::GlobalDirective) {
                global_directives.push(GlobalDirective::from_cst(cst, child))
            } else if cst.match_rule(child, Rule::GlobalDeclaration) {
                global_declarations.push(GlobalDeclaration::from_cst(cst, child).into())
            } else {
                panic!("unexpected node")
            }
        }

        TranslationUnit {
            imports,
            global_directives,
            global_declarations,
        }
    }
}

impl FromCst for GlobalDirective {
    fn from_cst(cst: &Cst, node: NodeRef) -> Self {
        println!("directive: {:?}", cst.get(node));
        todo!()
    }
}

impl FromCst for GlobalDeclaration {
    fn from_cst(cst: &Cst, node: NodeRef) -> Self {
        println!("decl: {:?}", cst.get(node));
        todo!()
    }
}
