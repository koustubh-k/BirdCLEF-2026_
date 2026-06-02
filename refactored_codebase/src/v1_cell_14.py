import pandas as pd, os, time, sys
from warnings import filterwarnings; filterwarnings("ignore")


##########################################################################################
##########################################################################################
#                                                                                        #
#           Additions and corrections by South Korean expert Pilkwang Kim                #
#                                                                                        #
#          https://www.kaggle.com/code/pilkwang/birdclef-2026-safe-ensemble              #
#                                                                                        #
##########################################################################################
##########################################################################################

def sinlge():
    pass

def _read_submission_checked(path):
    df = pd.read_csv(path)
    assert "row_id" in df.columns, f"row_id column missing in {path}"
    assert df["row_id"].is_unique, f"duplicate row_id values in {path}"
    prob_cols = [c for c in df.columns if c != "row_id"]
    assert prob_cols, f"no probability columns in {path}"
    values = df[prob_cols].to_numpy()
    assert np.isfinite(values).all(), f"NaN/inf values in {path}"
    assert values.min() >= 0.0 and values.max() <= 1.0, f"probabilities outside [0, 1] in {path}"
    return df.set_index("row_id")


def direct_add2():
    print(f'Ensemble: {_ensemble_models},   LB: {_lbs},   weights: {_weights}')
    assert len(_files_subm) == len(_weights), "submission file / weight length mismatch"
    weight_sum = float(sum(_weights))
    assert weight_sum > 0, "ensemble weights must sum to a positive value"
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        print(f"Normalizing ensemble weights from sum={weight_sum:.6f}")
    norm_weights = [float(w) / weight_sum for w in _weights]
    dfs = [_read_submission_checked(path) for path in _files_subm]
    base_idx = dfs[0].index
    base_cols = dfs[0].columns
    for path, df in zip(_files_subm, dfs):
        assert df.columns.equals(base_cols), f"Column mismatch in {path}"
        missing = base_idx.difference(df.index)
        extra = df.index.difference(base_idx)
        assert len(missing) == 0 and len(extra) == 0, (
            f"row_id mismatch in {path}: missing={len(missing)}, extra={len(extra)}"
        )
    out = sum(w * df.loc[base_idx, base_cols] for w, df in zip(norm_weights, dfs))
    values = out.to_numpy()
    assert np.isfinite(values).all(), "NaN/inf in final blend"
    assert values.min() >= 0.0 and values.max() <= 1.0, "final probabilities outside [0, 1]"
    return out
    

def direct_add_safe( solh ):
    
    _ensemble_models = [model['Model' ] for model in solh['Models']]
    _files_subm      = [model['subm'  ] for model in solh['Models']]
    _weights         = [model['weight'] for model in solh['Models']]
    _xsed            = [model['xSED'  ] for model in solh['Models']]
    _lbs             = [model['LB'    ] for model in solh['Models']]

    print(f'Ensemble: {_ensemble_models},   LB: {_lbs},   weights: {_weights}')
    assert len(_files_subm) == len(_weights), "submission file / weight length mismatch"
    weight_sum = float(sum(_weights))
    assert weight_sum > 0, "ensemble weights must sum to a positive value"
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        print(f"Normalizing ensemble weights from sum={weight_sum:.6f}")
    norm_weights = [float(w) / weight_sum for w in _weights]
    dfs = [_read_submission_checked(path) for path in _files_subm]
    base_idx = dfs[0].index
    base_cols = dfs[0].columns
    for path, df in zip(_files_subm, dfs):
        assert df.columns.equals(base_cols), f"Column mismatch in {path}"
        missing = base_idx.difference(df.index)
        extra = df.index.difference(base_idx)
        assert len(missing) == 0 and len(extra) == 0, (
            f"row_id mismatch in {path}: missing={len(missing)}, extra={len(extra)}"
        )
    out = sum(w * df.loc[base_idx, base_cols] for w, df in zip(norm_weights, dfs))
    values = out.to_numpy()
    assert np.isfinite(values).all(), "NaN/inf in final blend"
    assert values.min() >= 0.0 and values.max() <= 1.0, "final probabilities outside [0, 1]"
    return out

##########################################################################################
##########################################################################################

def take_of(solut, half):           # note:  ONLY__2__MODELS
    
    models = [{
      'Model' : model['Model'],
      'subm'  : model['subm'].replace('.csv',f'{i+1}.csv'), 
      'weight': model['wts_first_half'] if half==1 else model['wts_second_half'],
      'xSED'  : model['xSED'],
      'LB'    : model['LB']
    } for i,model in enumerate(solut['Models'])]
    
    solutions = {'type_add':solut['type_add'],'Models': models}

    _boundary = solut['task1'][f'boundary']

    df1       = pd.read_csv(solut['Models'][0]['subm'])
    df1_half  = df1.iloc[:,1:_boundary] if half==1 else df1.iloc[:,_boundary:]
    df1_rowId = df1.iloc[:,[0,1]]
    df1       = pd.concat([df1_rowId, df1_half], axis=1)
    df1.to_csv(solutions['Models'][0]['subm'],index=True)

    df2       = pd.read_csv(solut['Models'][1]['subm'])
    df2_half  = df2.iloc[:,1:_boundary] if half==1 else df2.iloc[:,_boundary:]
    df2_rowId = df2.iloc[:,[0,1]]
    df2       = pd.concat([df2_rowId, df2_half], axis=1)
    df2.to_csv(solutions['Models'][1]['subm'],index=True)
    
    return solutions
    

def direct_add2_safe():
    
    _boundary = None
    
    if 'task1' in solut:
        if 'div' in solut['task1']:
            if solut['task1']['div'] == 'dataframe_on_2_half':
                _boundary = solut['task1'][f'boundary']
                
    if _boundary != None:
        
        solh1 = take_of ( solut, half=1 )  ;out1 = direct_add_safe(solh1)
        solh2 = take_of ( solut, half=2 )  ;out2 = direct_add_safe(solh2)

        out = pd.merge(out1,out2, on='row_id')

    return out
    
##########################################################################################
##########################################################################################

def direct_safe():
    if len(_ensemble_models) == 2:  return direct_add2_safe()

    
def direct():
    if len(_ensemble_models) == 2:  return direct_add2()