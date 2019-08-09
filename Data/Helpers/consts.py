
import os

################################################################################
#################################### PATHS #####################################
################################################################################

if 'AVSRHOST' in os.environ and os.environ['AVSRHOST'] == 'IMS':
    LISTPATH = '/mount/arbeitsdaten34/projekte/thangvu/collaboration/MichaelW/data/GRID/FullDatabaseAllSpeakers'
    STATSDIR = '/mount/arbeitsdaten34/projekte/thangvu/collaboration/MichaelW/data/GRID/Stats.cleaned'
    DICTIONARYFILE = '/mount/arbeitsdaten34/projekte/thangvu/collaboration/MichaelW/data/GRID/FullDatabaseAllSpeakers/Dictionary.txt'
    PHONELISTFILE = '/mount/arbeitsdaten34/projekte/thangvu/collaboration/MichaelW/data/GRID/FullDatabaseAllSpeakers/PhoneList.txt'
    WORDLISTFILE = '/mount/arbeitsdaten34/projekte/thangvu/collaboration/MichaelW/data/GRID/FullDatabaseAllSpeakers/WordList.txt'
    #DataDir = '/mount/arbeitsdaten34/projekte/thangvu/collaboration/MichaelW/data/GRID/Corpus/%(spk)s'
    DATADIR = '/mount/arbeitsdaten34/projekte/thangvu/collaboration/MichaelW/data/GRID/Audio_Pickles/%(spk)s'
elif 'AVSRHOST' in os.environ and os.environ['AVSRHOST'] == 'CLUS':
    # this is the Lugano cluster
    LISTPATH = '/scratch/snx3000/mwand/projects/Audiovisual/GRID/FullDatabaseAllSpeakers'
    STATSDIR = '/scratch/snx3000/mwand/projects/Audiovisual/GRID/Stats'
    DICTIONARYFILE = LISTPATH + '/Dictionary.txt'
    PHONELISTFILE = LISTPATH + '/PhoneList.txt'
    WORDLISTFILE = LISTPATH + '/WordList.txt'
    DATADIR = '/scratch/snx3000/mwand/projects/Audiovisual/GRID/Corpus/%(spk)s'
else:
    LISTPATH = '/home/mwand/projects/AudioVisual/GRID-old/FullDatabaseAllSpeakers'
    STATSDIR = '/home/mwand/projects/AudioVisual/GRID-old/Stats'
    DIFFSTATSDIR = '/home/rivama/projects/lipreading/DiffStats'
    DICTIONARYFILE = '/home/mwand/projects/AudioVisual/GRID-old/FullDatabaseAllSpeakers/Dictionary.txt'
    PHONELISTFILE = '/home/mwand/projects/AudioVisual/GRID-old/FullDatabaseAllSpeakers/PhoneList.txt'
    WORDLISTFILE = '/home/mwand/projects/AudioVisual/GRID-old/FullDatabaseAllSpeakers/WordList.txt'
    DATADIR = '/home/mwand/projects/AudioVisual/GRID-old/Corpus/%(spk)s'

TRAIN_PATH = LISTPATH + '/train_sd_cop_shuffled.txt'
VALID_PATH = LISTPATH + '/dev_sd_cop.txt'
TEST_PATH = LISTPATH + '/eval_sd_cop.txt'

################################################################################
################################## CONSTANTS ###################################
################################################################################

VIDEO_INFIX = 'NewMouths-Grayscale-80x40-Type1'
TOTAL_MAX_FRAME = 75