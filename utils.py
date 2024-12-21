'''
tool functions
'''
import math
import datetime
import os

def prinT(str_to_print: str):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "--", str_to_print)

def get_data_dir():
    # get the path of the class file
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # construct the path of the data directory
    data_dir = os.path.join(file_dir, 'data')
    print("data_dir: ", data_dir)
    return data_dir
    
def pow10ceil(x):
    # 101, 500 -> 1000
    return 10**math.ceil(math.log(x, 10))

Scopus2color = {'Chemical Engineering': '#6C0000',
                'Chemistry': '#9A0000',
                'Computer Science': '#FF5C29',
                'Earth and Planetary Sciences': '#FE0000',
                'Energy': '#FF7C80',
                'Engineering': '#D20000',
                'Environmental Science': '#D26B04',
                'Materials Science': '#FC9320',
                'Mathematics': '#FBFF57',
                'Physics and Astronomy': '#FFCC00',
                
                'Medicine' :'#7030A0',
                'Nursing': '#9900CC',
                'Veterinary': '#CC00FF',
                'Dentistry': '#A679FF',
                'Health Professions': '#CCB3FF',

                'Arts and Humanities': '#375623',
                'Business, Management and Accounting': '#187402',
                'Decision Sciences': '#16A90F',
                'Economics, Econometrics and Finance': '#8FA329',
                'Psychology': '#92D050',
                'Social Sciences': '#66FF66',

                'Agricultural and Biological Sciences': '#000099',
                'Biochemistry, Genetics and Molecular Biology': '#336699',
                'Immunology and Microbiology': '#0000F2',
                'Neuroscience': '#0099FF',
                'Pharmacology, Toxicology and Pharmaceutics': '#85D6FF',

                'Multidisciplinary': '#000000'}

periodical_title_stopwords = ['the', 'and', 'of', 'to', 'in', 'a', 'for', 'is', 'on', 'that', 
                               'de', 'di', 'des', 'del',
                               'medizinische', 'clinica', 'wochenschrift', 'rendiconti',
                               'journal', 'letters', 'studies', 'international', 'transactions', 'acta', 
                               'revista', 'research', 'review', 
                               'sciences', 'science', 'chinese', 'theory', 'sinica', 'applied', 'applications']

# incorrect venues in MAG
corrupted_venue_name_list = ['Japanese Journal of Pharmacology', 
                             'Journal of Computers',
                             'Journal of Algorithms',
                             'Journal of Agricultural Engineering Research',
                             'Sozial-und Praventivmedizin',
                             'Scientia Forestalis',
                             'Interpretation',
                             'Genes',
                             'Protein Science',
                             'Hospital Medicine',
                             'Immunotechnology',
                             'Journal of Ayurveda and Integrative Medicine',
                             'Versus',
                             'Tradition',
                             'ACM Transactions on Cyber-Physical Systems',
                             'Journal of Biomedical Engineering',
                             'Antibiotics and Chemotherapy',
                             'Social Work',
                             'Production Journal',
                             'Insight',
                             'Sats',
                             'English',
                             'Leonardo',
                             'The Forum',
                             'Chemical Industry',
                             'The American review of respiratory disease',
                             'Chemistry & Industry',
                             'Lawrence Berkeley National Laboratory',
                             'international symposium on open collaboration',
                             'Scopus',
                             'Technical reports',
                             'The annual research report',
                             'Biosilico',
                             '0',
                             'Computer Science and Its Applications',
                             'Journal of Programming Languages',
                             'Fuel and Energy Abstracts',
                            ]