import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from type import Word

FIENDISH_WORDS = [
    Word(df_index=3367, hash=6234, word='emcee', occurrence=9.538214207793772e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=1428, hash=42882, word='bravo', occurrence=9.489698172160389e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=7413, hash=12105, word='nerdy', occurrence=9.483800494702877e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8977, hash=18684, word='rearm', occurrence=9.463796359909792e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=3324, hash=24265, word='eking', occurrence=9.455837059135774e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=4262, hash=38935, word='gawky', occurrence=9.397296494739749e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=11107, hash=5278, word='tapir', occurrence=9.37644338705468e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=1364, hash=19959, word='boule', occurrence=9.369705310291465e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=2128, hash=60305, word='cluck', occurrence=9.35717641681322e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=10247, hash=30654, word='slurp', occurrence=9.3087445485196e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=5543, hash=43232, word='jiffy', occurrence=9.283602345533381e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=3549, hash=7686, word='eying', occurrence=9.255129775453953e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=7663, hash=55305, word='odder', occurrence=9.25128374973383e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=10131, hash=47164, word='skimp', occurrence=9.249229840690986e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=4523, hash=44723, word='goner', occurrence=9.190305075179597e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=12450, hash=64793, word='whelp', occurrence=9.13640725030973e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8531, hash=26826, word='preen', occurrence=9.030641408003248e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=9258, hash=40441, word='roger', occurrence=9.028281070300181e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=4887, hash=50123, word='harpy', occurrence=8.999245544316639e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=1992, hash=27183, word='chump', occurrence=8.978836625317399e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=5238, hash=56333, word='hunky', occurrence=8.963035300624255e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=11894, hash=20286, word='unfed', occurrence=8.952564002129293e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=568, hash=42859, word='artsy', occurrence=8.888647005989014e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=31, hash=18866, word='abled', occurrence=8.85559484231635e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=2993, hash=46715, word='dopey', occurrence=8.763230809449852e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=1167, hash=23190, word='bleep', occurrence=8.753583163212399e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=9984, hash=21521, word='shush', occurrence=8.625239207304959e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=6441, hash=28603, word='loopy', occurrence=8.623006500840804e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=6606, hash=40499, word='macaw', occurrence=8.587821042738142e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=3926, hash=39697, word='flunk', occurrence=8.549681709268953e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=647, hash=52087, word='aunty', occurrence=8.417815692496335e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=9403, hash=21626, word='rumba', occurrence=8.39553034026608e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=10644, hash=15133, word='squib', occurrence=8.39347818271108e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=4267, hash=33595, word='gayly', occurrence=8.334735888126942e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=9128, hash=33038, word='retch', occurrence=8.327526241913577e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8469, hash=49237, word='poser', occurrence=8.26081433302761e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=7559, hash=46861, word='nosey', occurrence=8.213139690127494e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8499, hash=61635, word='pouty', occurrence=8.112289004991168e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=1290, hash=37331, word='bongo', occurrence=8.103535208192626e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=703, hash=31867, word='axion', occurrence=8.067932578459393e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=12214, hash=63010, word='voila', occurrence=8.065285740599394e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=9216, hash=9980, word='riper', occurrence=8.049073230154138e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=9005, hash=24241, word='recut', occurrence=7.817905611773315e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8862, hash=3463, word='rajah', occurrence=7.799040524503199e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=3340, hash=5162, word='elide', occurrence=7.725986126416729e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=6202, hash=49880, word='lefty', occurrence=7.717100460880036e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=7727, hash=28245, word='ombre', occurrence=7.702176048951515e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=2641, hash=27158, word='daunt', occurrence=7.689052509363137e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=9384, hash=48344, word='ruder', occurrence=7.680083648153868e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=1323, hash=35093, word='boozy', occurrence=7.547872005631006e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=10815, hash=63914, word='stunk', occurrence=7.48104102044067e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=2540, hash=5297, word='cutie', occurrence=7.47395973199616e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=2406, hash=7978, word='crick', occurrence=7.468983653780016e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=7480, hash=18127, word='ninny', occurrence=7.421384232486616e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=3751, hash=14041, word='ficus', occurrence=7.41610844556817e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=12899, hash=33945, word='zesty', occurrence=7.402563994673983e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=1354, hash=54958, word='botch', occurrence=7.392880306156259e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=2129, hash=50251, word='clued', occurrence=7.31754245109073e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8429, hash=48282, word='pooch', occurrence=7.309823969237073e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=1649, hash=59183, word='cabby', occurrence=7.27721118920499e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=12528, hash=11424, word='wimpy', occurrence=7.169004303975157e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8115, hash=15449, word='penne', occurrence=7.163120375963672e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8360, hash=26530, word='plunk', occurrence=7.142120352909845e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=11935, hash=48722, word='unset', occurrence=7.130362696727844e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=4742, hash=59092, word='guppy', occurrence=7.119867319715922e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=6672, hash=7901, word='mambo', occurrence=6.885137985079836e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=10916, hash=3081, word='swami', occurrence=6.851748572955787e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=10304, hash=12658, word='snaky', occurrence=6.83528439537895e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=6973, hash=25644, word='minty', occurrence=6.698928038062491e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8020, hash=18450, word='patsy', occurrence=6.65416113321271e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=5635, hash=16453, word='junto', occurrence=6.466443554842271e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=2846, hash=55569, word='dilly', occurrence=6.448898950850436e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=4284, hash=48408, word='geeky', occurrence=6.334700970063444e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8877, hash=29400, word='ramen', occurrence=6.239184536571685e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=3836, hash=21552, word='flack', occurrence=6.226244146034789e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=12, hash=55044, word='abase', occurrence=6.17710364991808e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=10158, hash=47549, word='skulk', occurrence=6.080900366356447e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8246, hash=51574, word='piney', occurrence=6.06935574154477e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=12065, hash=3484, word='vaunt', occurrence=5.8322153932977016e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=882, hash=27791, word='batty', occurrence=5.730399468717452e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=4266, hash=27763, word='gayer', occurrence=5.6712346889753474e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=11493, hash=64985, word='tonga', occurrence=5.4754059206629794e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=4475, hash=15580, word='gnash', occurrence=4.9668348651721314e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=3287, hash=57956, word='eclat', occurrence=4.8553799825157294e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=3329, hash=28160, word='elate', occurrence=4.795970291127105e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=6752, hash=36359, word='matey', occurrence=4.5903498069321816e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=4271, hash=44123, word='gazer', occurrence=4.491006578177803e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8334, hash=16755, word='plier', occurrence=4.40305071336411e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=5195, hash=61706, word='howdy', occurrence=4.290328373457441e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=12598, hash=30402, word='wooer', occurrence=4.2814162775073343e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=1049, hash=26234, word='biddy', occurrence=4.1442851212991634e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=1289, hash=64017, word='boney', occurrence=4.1431725712470784e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=8874, hash=59845, word='ralph', occurrence=4.040472775912463e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=4061, hash=48610, word='fritz', occurrence=3.293338648902022e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=12414, hash=9047, word='welsh', occurrence=3.0912513615533046e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=577, hash=49660, word='ascot', occurrence=2.8376055443146697e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=2456, hash=5401, word='crump', occurrence=2.370160146369926e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=12407, hash=13811, word='welch', occurrence=2.3429429116816893e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=5232, hash=21994, word='humph', occurrence=2.2163575055245133e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=9598, hash=62475, word='savoy', occurrence=2.0758936365439243e-08, linked_trajectories=[], games_won=0, games_total=0),
    Word(df_index=7974, hash=55338, word='parer', occurrence=1.4275635074056937e-08, linked_trajectories=[], games_won=0, games_total=0)
]