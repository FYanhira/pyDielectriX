from graphviz import Digraph

# Crear diagrama
dot = Digraph(comment="pyDipolAI Workflow", format="png")

# Ajustes de estilo
dot.attr(rankdir="TB", size="8")  # TB = top to bottom, también puedes usar LR = left to right
dot.attr('node', shape='box', style='rounded,filled', color='lightblue2', fontname="Helvetica")

# Nodos del workflow
dot.node("1", "Load experimental data\n(f, ε′r, ε″r)")
dot.node("2", "Select model(s) via GUI")
dot.node("3", "Initialize parameters")
dot.node("4", "Perform nonlinear fitting\n+ Bayesian optimization (optional)")
dot.node("5", "Evaluate fit quality\n(R², MSE, AAD)")
dot.node("6", "Apply FBN for ranking/selection")
dot.node("7", "Visualize results\n(ε*r and M*)")
dot.node("8", "Export outputs\n(publication/analysis)")

# Conexiones en orden
dot.edges([("1","2"), ("2","3"), ("3","4"), ("4","5"), 
           ("5","6"), ("6","7"), ("7","8")])

# Exportar
dot.render("workflow_pydipolai", view=True)
