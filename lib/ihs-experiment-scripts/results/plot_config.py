import matplotlib.pyplot as plt

plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Times"],
    "font.size" : 12,
    #"text.latex.preamble" : r"\usepackage{amsmath}"
})

# Take pallette from http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/#a-colorblind-friendly-palette
#             Grey,        Black,      Orange, Sky Blue,    Green,    Yellow,    Blue,      Orange,    Pink     Red
my_palette = ["#999999", "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",'#d7191c']


gauss_ihs_params = {
    'color'     : my_palette[-1],#'red',
    'linewidth' : 1.5,
    'linestyle' : ':',
    'marker'    : 'D',
    'markersize': 4.0
}

srht_ihs_params = {
    'color'     : my_palette[2],#'#fdae61',# 'magenta',
    'linewidth' : 1.5,
    'linestyle' : '-',
    'marker'    : '*',
    'markersize': 5.0
}

cnsk_ihs_params = {
    'color'     : my_palette[6],#'#2c7bb6', # 'royalblue',
    'linewidth' : 1.5,
    'linestyle' : '-.',
    'marker'    : '+',
    'markersize': 4.0
}

sjlt_ihs_params = {
    'color'     : my_palette[3], #'#abd9e9', # 'cyan',
    'linewidth' : 1.5,
    'linestyle' : '--',
    'marker'    : 'o',
    'markersize': 4.0
}

classical_params = {    
    'color'     : my_palette[8],#'goldenrod', #'#ffffbf', # 'forestgreen',
    'linewidth' : 1.5,
    'linestyle' : '-',
    'marker'    : 's',
    'markersize': 4.0
}

ihs_plot_params = {
    'Gaussian'   : gauss_ihs_params,
    'SRHT'       : srht_ihs_params,
    'CountSketch': cnsk_ihs_params,
    'SJLT'       : sjlt_ihs_params,
    'Classical'  : classical_params 
}


# ! For experiment 0
classical_plot_params = {
    'Classical CountSketch' : cnsk_ihs_params,
    'Classical SJLT'         : sjlt_ihs_params,
    'Classical SRHT'         : srht_ihs_params,
    'Classical Gaussian'     : gauss_ihs_params,
    'IHS CountSketch'        :{
        'label'     : 'IHS CountSketch',
        'color'     : my_palette[6],
        'linewidth' : 1.5,
        'linestyle' : '-.',
        'marker'    : '2',
        'markersize': 10.0
    },
    'IHS SJLT'              :{
        'label'     : 'IHS: SJLT',
        'color'     : my_palette[3], 
        'linewidth' : 1.5,
        'linestyle' : '--',
        'marker'    : '|',
        'markersize': 4.0
    },
   'IHS SRHT'              :{
        'label'     : 'IHS: SRHT',
        'color'     : my_palette[2],
        'linewidth' : 1.5,
        'linestyle' : '-',
        'marker'    : '_',
        'markersize': 5.0
    },
    'IHS Gaussian'          :{
        'label'     : 'IHS: Gaussian',
        'color'     : my_palette[-1],
        'linewidth' : 1.5,
        'linestyle' : ':',
        'marker'    : 'd',
        'markersize': 4.0
        },
    'Optimal'        :{
        'label'     : 'Optimal',
        'color'     : my_palette[4],
        'linewidth' : 1.0,
        'linestyle' : (0, (1, 10)),
        'marker'    : '>',
        'markersize': 10.0
    },
}


# ! Frequent Directions parameters
fd_synthetic_params = {
    'color' : 'grey',
    'linestyle' : ':',
}

rfd_synthetic_params = {
    'color' : 'black',
    'linestyle' : '-',
}

fd_experiment0_markers = {
    '-1' : {'marker' : 'o', 'markersize':5},
    '0'  : {'marker' : 'x', 'markersize':5},
    '1'  : {'marker' : '^', 'markersize':5},
    '2'  : {'marker' : 's', 'markersize':5},
}