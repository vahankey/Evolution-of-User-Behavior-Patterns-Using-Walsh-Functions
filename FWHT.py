def fwht(data):
    
    n = len(data)
    h = 1
    
    while h < n:
       
        for i in range(0, n, h * 2):
            
           
            for j in range(i, i + h):
                x = data[j]
                y = data[j + h]

                data[j] = x + y
                data[j + h] = x - y
                
        h *= 2 
        
    return data

if __name__ == "__main__":

    x = [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0]
    
    print("Ժամանակային վեկտոր (Լոգեր):", x)
    
    spectrum = fwht(x)
    
    n = len(spectrum)
    normalized_spectrum = [val // n for val in spectrum]
    
    print("Սպեկտրալ վեկտոր (Արդյունք):", normalized_spectrum)
