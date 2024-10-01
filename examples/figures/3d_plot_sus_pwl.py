from examples.numerical_examples.pwl.pwl import pwl
from nsus.implementation import SubsetSimulation
from examples.figures.utils import sus_3d_plot, classifier_contour_points
import numpy as np

#######################

sus = SubsetSimulation(performance_function=pwl,
                       dimension=2,
                       level_size=500,
                       threshold=0,
                       level_probability=0.1,
                       seed=5,
                       scale=1,
                       verbose=False)

sus.run()

#######################

def pwl_classifier(x):
    perf = pwl(x)
    return -1 if perf<0 else 1

contour_points = classifier_contour_points(pwl_classifier,
                                           (-4,6),
                                           (-4,6),
                                           0.01)
lss_x = contour_points[0].T[0]
lss_y = contour_points[0].T[1]
lss_z = np.array([pwl(pt) for pt in contour_points[0]])
lss_list = [(lss_x,lss_y,lss_z)]

########################

for i in [1,2,3,7]:
    legend = True if i == 7 else False
    sus_3d_plot(x_range=(-4,6),
                y_range=(-4,6),
                step=0.1,
                current_level=i,
                sus=sus,
                lss_list=lss_list,
                design_point=[4,0],
                partition_boundaries=[],
                path='examples/figures/images/pwl_3d_sus',
                legend=legend)




















