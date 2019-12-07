var N=null,E="",T="t",U="u",searchIndex={};
var R=["Calculation of the Jacobian J(x) of a vector function `fs`…","Calculation of the product of the Jacobian J(x) of a…","perturbationvectors","Calculation of the product of the Hessian H(x) of a…","result","perturbationvector","PerturbationVector","FiniteDiff"];

searchIndex["finitediff"]={"doc":"This crate contains a wide range of methods for the…","i":[[3,R[6],"finitediff","Perturbation Vector for the accelerated computation of the…",N,N],[12,"x_idx",E,"x indices",0,N],[12,"r_idx",E,"correspoding function indices",0,N],[11,"new",E,"Create a new empty `PerturbationVector`",0,[[],["self"]]],[11,"add",E,"Add an index `x_idx` and the corresponding function…",0,[[["vec",["usize"]],["usize"]],["self"]]],[6,"PerturbationVectors",E,"A collection of `PerturbationVector`s",N,N],[8,R[7],E,E,N,N],[16,"Jacobian",E,E,1,N],[16,"Hessian",E,E,1,N],[16,"OperatorOutput",E,E,1,N],[10,"forward_diff",E,"Forward difference calculated as",1,[[["self"],["fn"]],["self"]]],[10,"central_diff",E,"Central difference calculated as",1,[[["self"],["fn"]],["self"]]],[10,"forward_jacobian",E,R[0],1,[[["self"],["fn"]]]],[10,"central_jacobian",E,R[0],1,[[["self"],["fn"]]]],[10,"forward_jacobian_vec_prod",E,R[1],1,[[["self"],["fn"]],["self"]]],[10,"central_jacobian_vec_prod",E,R[1],1,[[["self"],["fn"]],["self"]]],[10,"forward_jacobian_pert",E,E,1,[[["self"],["fn"],[R[2]]]]],[10,"central_jacobian_pert",E,E,1,[[["self"],["fn"],[R[2]]]]],[10,"forward_hessian",E,"Calculation of the Hessian using forward differences",1,[[["self"],["fn"]]]],[10,"central_hessian",E,"Calculation of the Hessian using central differences",1,[[["self"],["fn"]]]],[10,"forward_hessian_vec_prod",E,R[3],1,[[["self"],["fn"]],["self"]]],[10,"central_hessian_vec_prod",E,R[3],1,[[["self"],["fn"]],["self"]]],[10,"forward_hessian_nograd",E,"Calculation of the Hessian using forward differences…",1,[[["self"],["fn"]]]],[10,"forward_hessian_nograd_sparse",E,"Calculation of a sparse Hessian using forward differences…",1,[[["self"],["fn"],["vec"]]]],[11,"to_owned",E,E,0,[[["self"]],[T]]],[11,"clone_into",E,E,0,[[["self"],[T]]]],[11,"into",E,E,0,[[],[U]]],[11,"from",E,E,0,[[[T]],[T]]],[11,"try_from",E,E,0,[[[U]],[R[4]]]],[11,"try_into",E,E,0,[[],[R[4]]]],[11,"borrow_mut",E,E,0,[[["self"]],[T]]],[11,"borrow",E,E,0,[[["self"]],[T]]],[11,"type_id",E,E,0,[[["self"]],["typeid"]]],[11,"default",E,E,0,[[],[R[5]]]],[11,"clone",E,E,0,[[["self"]],[R[5]]]]],"p":[[3,R[6]],[8,R[7]]]};
initSearch(searchIndex);addSearchOptions(searchIndex);