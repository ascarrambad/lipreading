
from Logger.BatchExperiment import BatchExperiment

SPEAKERS_COUPLES = [
    's1_s2',
    's2_s3',
    's3_s4',
    's4_s5',
    's9_s10',
    's10_s11',
    's11_s12',
    's12_s13',
    's13_s14',
    's14_s15',
    's15_s16',
    's16_s17',
    's17_s18',
    's18_s19',
    's19_s20',
]

SPEAKERS_FOUR = [
's1-s2-s3-s4_s5',
's2-s3-s4-s5_s6',
's3-s4-s5-s6_s7',
's4-s5-s6-s7_s8',
]

SPEAKERS_EIGHT = [
's12-s13-s14-s15-s16-s17-s18-s19_s20',
's13-s14-s15-s16-s17-s18-s19-s20_s1',
's14-s15-s16-s17-s18-s19-s20-s1_s2',
's15-s16-s17-s18-s19-s20-s1-s2_s3',
's16-s17-s18-s19-s20-s1-s2-s3_s4',
's17-s18-s19-s20-s1-s2-s3-s4_s5',
's18-s19-s20-s1-s2-s3-s4-s5_s6',
's19-s20-s1-s2-s3-s4-s5-s6_s7',
's20-s1-s2-s3-s4-s5-s6-s7_s8',
]

SPEAKERS_SIXTEEN = [
's1-s2-s3-s4-s5-s6-s7-s8-s9-s10-s11-s12-s13-s14-s15-s16_s17',
's2-s3-s4-s5-s6-s7-s8-s9-s10-s11-s12-s13-s14-s15-s16-s17_s18',
's3-s4-s5-s6-s7-s8-s9-s10-s11-s12-s13-s14-s15-s16-s17-s18_s19',
's4-s5-s6-s7-s8-s9-s10-s11-s12-s13-s14-s15-s16-s17-s18-s19_s20',
's5-s6-s7-s8-s9-s10-s11-s12-s13-s14-s15-s16-s17-s18-s19-s20_s1',
's6-s7-s8-s9-s10-s11-s12-s13-s14-s15-s16-s17-s18-s19-s20-s1_s2',
's7-s8-s9-s10-s11-s12-s13-s14-s15-s16-s17-s18-s19-s20-s1-s2_s3',
's8-s9-s10-s11-s12-s13-s14-s15-s16-s17-s18-s19-s20-s1-s2-s3_s4',
's9-s10-s11-s12-s13-s14-s15-s16-s17-s18-s19-s20-s1-s2-s3-s4_s5',
's10-s11-s12-s13-s14-s15-s16-s17-s18-s19-s20-s1-s2-s3-s4-s5_s6',
's11-s12-s13-s14-s15-s16-s17-s18-s19-s20-s1-s2-s3-s4-s5-s6_s7',
's12-s13-s14-s15-s16-s17-s18-s19-s20-s1-s2-s3-s4-s5-s6-s7_s8',
's13-s14-s15-s16-s17-s18-s19-s20-s1-s2-s3-s4-s5-s6-s7-s8_s9',
's14-s15-s16-s17-s18-s19-s20-s1-s2-s3-s4-s5-s6-s7-s8-s9_s10',
's15-s16-s17-s18-s19-s20-s1-s2-s3-s4-s5-s6-s7-s8-s9-s10_s11',
's16-s17-s18-s19-s20-s1-s2-s3-s4-s5-s6-s7-s8-s9-s10-s11_s12',
's17-s18-s19-s20-s1-s2-s3-s4-s5-s6-s7-s8-s9-s10-s11-s12_s13',
's18-s19-s20-s1-s2-s3-s4-s5-s6-s7-s8-s9-s10-s11-s12-s13_s14',
's19-s20-s1-s2-s3-s4-s5-s6-s7-s8-s9-s10-s11-s12-s13-s14_s15',
's20-s1-s2-s3-s4-s5-s6-s7-s8-s9-s10-s11-s12-s13-s14-s15_s16',
]

Params = {'AllSpeakers': SPEAKERS_COUPLES, 'TensorboardDir': None}

db_path = 'mongodb+srv://lab:alchemist94@experiments-xa6jh.mongodb.net/test?retryWrites=true&w=majority'
scripts = ['std_fc', 'dyn_net', 'mcnet_fc']

exp = BatchExperiment(script_names=scripts, gpus=[0,1,2,3], exp_params=Params, db_path=db_path)
exp.run()

print('Done.')
