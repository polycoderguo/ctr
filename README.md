## Originally from https://github.com/breakhearts/ctr

# This is a final project from SJSU 

### Goal:
To predict click through rate for a given dataset. 

### Procedure:
1. Feature engineering 
   1. separate data and site data using id/domain/category
   2. use pub_category, pub_domain, pub_id
2. Model training
   1. tune epochs
   2. add early stopping
   3. add feature which enable dumping model training artifacts out 
   4. enable model warm start training. 
