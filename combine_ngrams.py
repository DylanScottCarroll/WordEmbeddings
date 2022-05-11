import string, time

def iterate_sections(start_point = None):
    i_start, j_start = start_point or ('a' , 'a')
    i_start, j_start = ord(i_start), ord(j_start)

    end = ord('z')+1

    for i in range(i_start, end):
        for j in range(j_start, end):
            yield chr(i), chr(j), (((i-ord('a'))*26) + (j-ord('a')) + 1)

def sum_into_dict(dict: dict, line: str) -> None:
    key, value = line.split(", ")
    value = int(value)

    l_key = key.lower()
    dict[l_key] = dict.get(l_key, 0) + value

def print_elapsed_time(start_time:float) -> None:
    t = time.time() - start_time

    hours   = t/(60*60)
    minutes = (hours-int(hours))*60
    seconds = (minutes-int(minutes))*60


    print("Elapsed Time - %02d:%02d:%02d" %(hours, minutes, seconds))

def main():    
    start_time:float = time.time()
    
    for a in string.ascii_lowercase:
        print(f"Reading the \"{a}\" file...")

        dest_file: str = "combined_ngram_corpus/" + a + ".txt"

        dest_dict: dict = {}      
        for b in string.ascii_lowercase:
            percent:float = 100 * ((ord(a)-ord("a"))*26 + (ord(b)-ord("a"))) / (26*26)
            print(f"\tLoading \"{a+b}\" ({percent :0.2f}%)" )

            src_file : str = "ngram_corpus/" + a + b + ".txt"
            
            try:
                with open(src_file, "r") as src: 
                    for line in src:
                        sum_into_dict(dest_dict, line)

            except FileNotFoundError:
                continue

        print(f"Writing the \"{a}\" file...\n")
        with open(dest_file, "w") as dest:
            for key, value in dest_dict.items():
                dest.write(key + ", " + str(value) + "\n")
        
        print_elapsed_time(start_time)


    


if __name__ == "__main__":
    main()