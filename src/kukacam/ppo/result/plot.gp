set xlabel 'Seasons'
set ylabel 'Mean scores'
set key inside right bottom
set title 'Effect of Attenuation on PPO training'
set grid
set xrange [0:25]
plot 'result_ppo_clip.txt' u 1:3 w l lw 3 lt 7 t 'PPO w/o attn', \
       'mh_attn_2.txt' u 1:3 w l lw 2 t 'MH Attn w 2 heads', \
       'mh_attn_4.txt' u 1:3 w l lw 2 t 'MH Attn w 4 heads', \
       'luang_attn_arch_1.txt' u 1:3 w l lw 2 t 'Luong-arch-1', \
       'luang_attn_arch_2.txt' u 1:3 w l lw 2 t 'Luong-arch-2', \
           'luong-arch-3.txt' u 1:3 w l lw 2 dt 4 t 'Luong-arch-3',\
         'bahdanau-arch-1.txt' u 1:3 w l lw 2 dt 3 t 'Bahdanau-arch-1',\
     'bahdanau-arch-2-3.txt' u 1:3 w l lw 2 t 'Bahdanau-arch-2',\
       'bahdanau-arch-3.txt' u 1:3 w l lw 2 dt 2 t 'Bahdanau-arch-3'
#     'bahdanau-arch-2-3.txt' u 1:3 w l lw 2 dt 4 t 'Bahdanau-arch-2-3',\
#     'bahdanau-attn-arch-2.txt' u 1:3 w l lw 2 t 'Bahdanau-arch-2-2'
#'bahdanau-arch-2-2layers-1.txt' u 1:3 w l lw 2 t 'Bahdanau-arch-2-2layers', \
#'bahdanau-arch-3-2layers.txt' u 1:3 w l lw 2 dt 3 t 'Bahdanau-arch-3-2layers',\
#           'luong-arch-3-2layers.txt' u 1:3 w l lw 2 dt 5 t 'Luong-arch-3-2layers'

