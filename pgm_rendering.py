from matplotlib import rc
rc("font", family="serif", size=8)
rc("text", usetex=True)

import daft

# Instantiate the PGM.
pgm = daft.PGM([8., 8.], origin=[0.3, 0.3])

# Hierarchical parameters.
pgm.add_node(daft.Node("t1k", r"$T_1$(k)", 3., 6.,scale = 1.3))
pgm.add_node(daft.Node("t1k-1", r"$T_1$(k-1)", 1., 6.,scale = 1.3))
pgm.add_node(daft.Node("p1k", r"$P_1$(k)", 3., 7.,scale = 1.3))
pgm.add_node(daft.Node("p1k-1", r"$P_1$(k-1)", 1., 7.,scale = 1.3))
pgm.add_node(daft.Node("t2k", r"$T_2$(k)", 3., 2.,scale = 1.3))
pgm.add_node(daft.Node("t2k-1", r"$T_2$(k-1)", 1., 2.,scale = 1.3))
pgm.add_node(daft.Node("p2k", r"$P_2$(k)", 3., 1.,scale = 1.3))
pgm.add_node(daft.Node("p2k-1", r"$P_2$(k-1)", 1., 1.,scale = 1.3))
pgm.add_node(daft.Node("ik", r"$I$(k)", 3., 4.,scale = 1.3))
pgm.add_node(daft.Node("f1k", r"$L_1$(k)", 3.4, 5.,scale = 1.3))
pgm.add_node(daft.Node("f2k", r"$L_2$(k)", 3.4, 3.,scale = 1.3))

pgm.add_node(daft.Node("t1k+1", r"$T_1$(k+1)", 5., 6.,scale = 1.3))
pgm.add_node(daft.Node("p1k+1", r"$P_1$(k+1)", 5., 7.,scale = 1.3))
pgm.add_node(daft.Node("t2k+1", r"$T_2$(k+1)", 5., 2.,scale = 1.3))
pgm.add_node(daft.Node("p2k+1", r"$P_2$(k+1)", 5., 1.,scale = 1.3))
pgm.add_node(daft.Node("ik+1", r"$I$(k+1)", 5., 4.,scale = 1.3))
pgm.add_node(daft.Node("f1k+1", r"$L_1$(k+1)", 5.4, 5.,scale = 1.3))
pgm.add_node(daft.Node("f2k+1", r"$L_2$(k+1)", 5.4, 3.,scale = 1.3))

pgm.add_node(daft.Node("t1k+2", r"$T_1$(k+2)", 7., 6.,scale = 1.3))
pgm.add_node(daft.Node("p1k+2", r"$P_1$(k+2)", 7., 7.,scale = 1.3))
pgm.add_node(daft.Node("t2k+2", r"$T_2$(k+2)", 7., 2.,scale = 1.3))
pgm.add_node(daft.Node("p2k+2", r"$P_2$(k+2)", 7., 1.,scale = 1.3))
pgm.add_node(daft.Node("ik+2", r"$I$(k+2)", 7., 4.,scale = 1.3))
pgm.add_node(daft.Node("f1k+2", r"$L_1$(k+2)", 7.4, 5.,scale = 1.3))
pgm.add_node(daft.Node("f2k+2", r"$L_2$(k+2)", 7.4, 3.,scale = 1.3))

# pgm.add_node(daft.Node("alpha", r"$\a$", 0.5, 2, fixed=True))
# pgm.add_node(daft.Node("beta", r"$\beta$", 1.5, 2))
# pgm.add_node(daft.Node("gamma", r"$\gamma$", 1.1, 1.))

# Latent variable.
# pgm.add_node(daft.Node("w", r"$w_n$", 1, 1))

# Data.
# pgm.add_node(daft.Node("x", r"$x_n$", 2, 1, observed=True))

# Add in the edges.
pgm.add_edge("t1k", "ik")
pgm.add_edge("t1k-1", "ik")
pgm.add_edge("t1k-1", "t1k")
pgm.add_edge("p1k-1", "t1k")
pgm.add_edge("p1k", "t1k")
pgm.add_edge("t1k-1", "t1k")
pgm.add_edge("t1k", "f1k")

pgm.add_edge("t2k", "ik")
pgm.add_edge("t2k-1", "ik")
pgm.add_edge("t2k-1", "t2k")
pgm.add_edge("p2k-1", "t2k")
pgm.add_edge("p2k", "t2k")
pgm.add_edge("t2k-1", "t2k")
pgm.add_edge("t2k", "f2k")

pgm.add_edge("t1k+1", "ik+1")
pgm.add_edge("t1k", "ik+1")
pgm.add_edge("p1k+1", "t1k+1")
pgm.add_edge("t1k", "t1k+1")
pgm.add_edge("p1k", "t1k+1")
pgm.add_edge("t1k+1", "f1k+1")

pgm.add_edge("t2k+1", "ik+1")
pgm.add_edge("t2k", "ik+1")
pgm.add_edge("p2k+1", "t2k+1")
pgm.add_edge("t2k", "t2k+1")
pgm.add_edge("p2k", "t2k+1")
pgm.add_edge("t2k+1", "f2k+1")

pgm.add_edge("t1k+2", "ik+2")
pgm.add_edge("p1k+2", "t1k+2")
pgm.add_edge("t1k+2", "f1k+2")
pgm.add_edge("t1k+1", "ik+2")
pgm.add_edge("t1k+1", "t1k+2")
pgm.add_edge("p1k+1", "t1k+2")

pgm.add_edge("t2k+2", "ik+2")
pgm.add_edge("p2k+2", "t2k+2")
pgm.add_edge("t2k+2", "f2k+2")
pgm.add_edge("t2k+1", "ik+2")
pgm.add_edge("t2k+1", "t2k+2")
pgm.add_edge("p2k+1", "t2k+2")

# And a plate.
# pgm.add_plate(daft.Plate([0.5, 0.5, 2, 1], label=r"$n = 1, \cdots, N$",
#     shift=-0.1))

# Render and save.
pgm.render()
pgm.figure.savefig("classic.pdf")
pgm.figure.savefig("classic.png", dpi=150)