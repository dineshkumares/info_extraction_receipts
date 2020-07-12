def get_text_features(df): 

    data = df['Object'].tolist()
    
    '''
        Args:
            str, input data
            
        Returns: 
            np.array, shape=(22,);
            an array of the text converted to features
            
    '''
    special_chars = {'&': 0, '@': 1, '#': 2, '(': 3, ')': 4, '-': 5, '+': 6, 
                    '=': 7, '*': 8, '%': 9, '.':10, ',': 11, '\\': 12,'/': 13, 
                    '|': 14, ':': 15}
    
    # character wise
    n_lower, n_upper, n_spaces, n_alpha, n_numeric,n_special = [],[],[],[],[],[]
    for words in data:
        upper,lower,alpha,spaces,numeric,special = 0,0,0,0,0,0
        for char in words: 
            # for lower letters 
            if char.islower(): 
                lower += 1
    
            # for upper letters 
            if char.isupper(): 
                upper += 1
            
            # for white spaces
            if char.isspace():
                spaces += 1
            
            # for alphabetic chars
            if char.isalpha():
                alpha += 1
            
            # for numeric chars
            if char.isnumeric():
                numeric += 1
                           
            if char in special_chars.keys():
                special += 1 

        n_lower.append(lower)
        n_upper.append(upper)
        n_spaces.append(spaces)
        n_alpha.append(alpha)
        n_numeric.append(numeric)
        n_special.append(special)
        #features.append([n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_digits])

    df['n_upper'],df['n_lower'],df['n_alpha'],df['n_spaces'],\
    df['n_numeric'],df['n_special'] = n_lower, n_upper, n_spaces, n_alpha, n_numeric,n_special

    print(df)
    return df 
