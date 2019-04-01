from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.continuous import LinearGaussianCPD

dbn = DBN()
dbn.add_edges_from([(('D', 0),('G', 0)),(('I', 0),('G', 0)),(('D', 0),('D', 1)),(('I', 0),('I', 1))])
grade_cpd = TabularCPD(('G', 0), 3, [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.8, 0.03], [0.3, 0.7, 0.02, 0.2]], evidence=[('I', 0),('D', 0)], evidence_card=[2, 2])
print 'grade_cpd', grade_cpd
d_i_cpd = TabularCPD(('D',1), 2, [[0.6, 0.3], [0.4, 0.7]], evidence=[('D',0)], evidence_card=[2])
diff_cpd = TabularCPD(('D', 0), 2, [[0.6, 0.4]])
intel_cpd = TabularCPD(('I', 0), 2, [[0.7, 0.3]])
i_i_cpd = TabularCPD(('I', 1), 2, [[0.5, 0.4], [0.5, 0.6]], evidence=[('I', 0)], evidence_card=[2])

dbn.add_edges_from([(('T', 0),('T',1)), (('P', 0), ('T', 1)), (('P', 1), ('T', 1)), (('T', 0), ('I', 1)), (('T', 1), ('I', 1)), (('E', 1), ('I', 1))])
trust_cpd = LinearGaussianCPD('T',  [0.2, -2, 3, 7], 9.6, ['T', 'P', 'X3'])
# interaction_cpd =
# extranous_interaction_cpd =





dbn.add_cpds(grade_cpd, d_i_cpd, diff_cpd, intel_cpd, i_i_cpd)
dbn.get_cpds()
t = dbn.get_interface_nodes(time_slice=0)
leaves = dbn.get_roots()
print t, leaves
