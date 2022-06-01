##########################

###### INPUT VARS ########

##########################



DATASETS_DIR = '/Users/sunhui/Desktop/ESPEI-NB-NI/ESPEI/input_test'

comps = ['NB', 'NI']

independent_component = 'NB'

OUTPUT_EXP_FILENAME = 'my_test_data.exp'



############################

############# RUN ##########

############################





import tinydb

from pycalphad import Database

from espei.datasets import load_datasets, recursive_glob

from espei.core_utils import ravel_zpf_values

from espei.utils import bib_marker_map





# load the experimental and DFT datasets

datasets = load_datasets(recursive_glob(DATASETS_DIR, '*.json'))

# phases = ['LIQUID', 'BCC_A2', 'FCC_A1']









desired_data = datasets.search((tinydb.where('output') == 'ZPF') &

                               (tinydb.where('components').test(lambda x: set(x).issubset(comps + ['VA']))) #&

                              # (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))

                              )







raveled_dict = ravel_zpf_values(desired_data, [independent_component])





bib_reference_keys = sorted(list({entry['reference'] for entry in desired_data}))

symbol_map = bib_marker_map(bib_reference_keys)
print(symbol_map)

# map matplotlib string markers to strings of markers for Thermo-Calc's POST

dataplot_symbols = ['S'+str(i) for i in range(1, 18)]

dataplot_marker_map = dict(zip([v['markers']['marker'] for v in symbol_map.values()], dataplot_symbols))
print('data=',dataplot_marker_map)






equilibria_to_plot = raveled_dict.get(2, [])


equilibria_lines = []
y_value=0.95
x_value=0.40
ref_key_list=[]
n=0
t_type=''
for eq in equilibria_to_plot:

    x_points, y_points = [], []
    for phase_name, comp_dict, ref_key in eq:
        sym_ref = symbol_map[ref_key]
        x_val, y_val = comp_dict[independent_component], comp_dict['T']
        if n>0:
            if ref_key not in ref_key_list:
                refer=str(x_value)+'\t'+str(y_value)+'\t'+"mna'"+'\t'+ref_key_list[n-1]+'\n'
                equilibria_lines.append(refer)
                y_value=int((y_value-0.05)*100)/100
        if n==(len(equilibria_to_plot)*2-1):
              refer=str(x_value)+'\t'+str(y_value)+'\t'+"mna'"+'\t'+ref_key+'\n'
              equilibria_lines.append(refer)
              y_value=int((y_value-0.05)*100)/100
        if t_type=='C':
            y_val=y_val-273
        if x_val is not None and y_val is not None:

            line = "{} {} {}".format(x_val, y_val, dataplot_marker_map[sym_ref['markers']['marker']])

            equilibria_lines.append(line)

        ref_key_list.append(ref_key)
        n=n+1






exp_file_lines = """$DATAPLOT Phase diagram, automatically generated

PROLOG 1 EXAMPLE 1  0<X<1, 300<Y<2500

DATASET 1 Two lines started with two symbols

ATTRIBUTE CENTER

BLOCK X=C1; Y=C2; GOC=C3,SWAS

""".splitlines()

exp_file_lines.extend(equilibria_lines)
# 0.55  0.85  mna'  Bennedek

exp_file_lines.append('BLOCK_END')


with open(OUTPUT_EXP_FILENAME,'w') as fp:

    fp.write("\n".join(exp_file_lines))