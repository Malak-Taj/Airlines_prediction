from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder


def preprocessing_pipeline():
    # Numerical Pipeline
    robust = RobustScaler()
    num_pipe = Pipeline( steps= [('scaling' , robust)])

    # Categorical Pipeline
    ohe = OneHotEncoder(drop = 'first' , sparse_output= False)
    ohe_Pipeline = Pipeline(steps = [('ohe' , ohe)])

    # Ordinal Pipeline 
    ordinal_encoder = OrdinalEncoder(
        categories=[['non-stop', '1 stop', '2 or more']], dtype=int)
    ordinal_pipe = Pipeline(steps=([('ordinal_encoder', ordinal_encoder)]))

    # Apply column Transformer
    preprocessing = ColumnTransformer(transformers = 
                                    ( [ ('Num_trans' , num_pipe ,['Duration', 'Dep_Hour']) , 
                                    ('hot_trans' , ohe_Pipeline , ['Airline', 'Source', 'Destination','Journey_Day','Journey_Month']),
                                    ('ord_trans' , ordinal_pipe , ['Total_Stops']) ]))
    return preprocessing