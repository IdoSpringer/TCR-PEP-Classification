import pstats

file = 'profile_2pep'

p = pstats.Stats(file)
p.sort_stats('cumulative').print_stats(100)