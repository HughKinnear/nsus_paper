import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



colors = [
      '#e66101',
      '#fdb863',
      '#b2abd2',
      '#5e3c99'
   ]


def contour_plot(x_range, y_range, step, function, color='black', levels=None):
    x = np.arange(x_range[0], x_range[1], step)
    y = np.arange(y_range[0], y_range[1], step)
    xx, yy = np.meshgrid(x, y)
    flat = np.vstack([xx.ravel(), yy.ravel()])
    z = np.array([function(point) for point in flat.T]).reshape(xx.shape)
    plt.contour(xx, yy, z, levels=levels,colors=color)


def plot_level(level_num,ss,color):
    levels = ss.all_levels[:level_num]
    for level in levels:
        plot_array = np.array([samp.array for samp in level.sample_list])[::5]
        plt.scatter(plot_array[:,0],plot_array[:,1],c=color,s=15)



def eval_3d(func,x_range,y_range,step):
    x = np.arange(x_range[0], x_range[1], step)
    y = np.arange(y_range[0], y_range[1], step)
    xx, yy = np.meshgrid(x, y)
    flat = np.vstack([xx.ravel(), yy.ravel()])
    z = np.array([func(point) for point in flat.T]).reshape(xx.shape)
    return xx,yy,z

def classifier_contour_points(classifier,x_range,y_range,step):
    xx,yy,z = eval_3d(classifier,x_range,y_range,step)
    contour = plt.contour(xx, yy, z)
    plt.close()
    return [path.vertices for path in contour.get_paths()]

def plot_3d_level(ax,level,marker_size,zorder,label,color):
    re_level = level.sample_list[0::10]
    array_level = [samp.array for samp in re_level]
    samp_x = np.array(array_level).T[0]
    samp_y = np.array(array_level).T[1]
    samp_z = np.array([samp.performance for samp in re_level])
    ax.scatter(samp_x,
               samp_y,
               samp_z,
               alpha=1,
               s=marker_size,
               color=color,
               marker='.',
               linewidth=0.2,
               edgecolor='black',
               label= label,
               zorder=zorder)
    
def sus_3d_plot(x_range,
                y_range,
                step,
                current_level,
                sus,
                lss_list,
                design_point,
                partition_boundaries,
                path,
                legend):
   
   

   performance_function = sus.performance_function.non_cache_performance_function
    
   tick_font_size = 10
   legend_font_size = 10
   xy_label_pad = 9
   z_label_pad = 5
   z_label_size = 15
   tick_pad = 4
   scatter_size = 100
   marker_size = 50
   xy_label_size = 20
   bound_zorder = 1
   lss_zorder = 2
   des_zorder = 3
   prev_zorder = 4
   curr_zorder = 5
   stride = 4
   lw = 0.05
   alpha = 1
   alias = True

   xx,yy,z = eval_3d(performance_function,x_range,y_range,step)

   plt.figure(figsize=(10, 7))
   ax = plt.axes(projection='3d',computed_zorder=False)
   ax.view_init(30,-120)
   plt.xlabel(u'x\u2081',labelpad=xy_label_pad,size=xy_label_size)
   plt.ylabel(u'x\u2082',labelpad=xy_label_pad,size=xy_label_size)
   ax.set_proj_type('ortho')
   ax.grid(False)
   # ax.set_zlabel('Performance',fontsize=z_label_size ,rotation=90,labelpad=z_label_pad)
   ax.tick_params(labelsize=tick_font_size,pad=tick_pad)
   ax.zaxis.set_ticks([-1,0,1,2])
   ax.zaxis.set_tick_params(pad=-14) 

   ax.plot_surface(xx,
                     yy,
                     z,
                     rstride=stride,
                     cstride=stride,
                     antialiased=alias,
                     color=colors[2],
                     alpha=alpha,
                     edgecolors='white', lw=lw)
   
   for i in range(current_level):
      level = sus.all_levels[i]
      if i == current_level-1:
         label = 'Current Level'
         zorder = curr_zorder
         color = colors[1]
      else:
         zorder = prev_zorder
         color = colors[0]
         label = 'Previous Levels' if i == 0 else None
      plot_3d_level(ax,level,scatter_size,zorder,label,color)

   ax.scatter(design_point[0],
              design_point[1],
              performance_function(design_point),
              alpha=1,
              s=marker_size,
              marker = 's',
              linewidth=0.5,
              color=colors[3],
              label='Design point',
              edgecolor='black',
              zorder=des_zorder
              )
   
   for lss in lss_list:
      ax.plot3D(lss[0],
                lss[1],
                lss[2],
                alpha=1,
                color=colors[3],
                linestyle = 'dashed',
                label='Limit state surface',
                zorder=lss_zorder
                )
      
   for boundary in partition_boundaries:
      ax.plot3D(boundary[0],
                boundary[1],
                boundary[2],
                alpha=1,
                linewidth=1,
                color='black',
                linestyle = 'solid',
                label='Partition boundary',
                zorder=bound_zorder
                )

   ax.text2D(0.05, 0.8, 'Performance', rotation=17, fontsize=14,  transform=ax.transAxes) 

   plt.savefig(path + f"_{str(current_level)}.pdf", bbox_inches='tight')


   if legend:
        fig_legend = plt.figure(figsize=(5, 1)) 
        legend = fig_legend.legend(*ax.get_legend_handles_labels(),ncols=2)
        fig_legend.canvas.draw()
        fig_legend.savefig(path + "_legend.pdf",bbox_inches='tight')
   
   plt.show()
   