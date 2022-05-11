import gzip, re, time, shutil, os, sys
from urllib import request, error


VALIDATION_REGEX = "^((:?[a-zA-Z]+\s){4}[a-zA-Z]+)\s([0-9]+)\s([0-9]+)\s([0-9]+)"

def iterate_sections(start_point):
    i_start, j_start = start_point or ('a' , 'a')
    i_start, j_start = ord(i_start), ord(j_start)

    end = ord('z')+1

    for i in range(i_start, end):
        for j in range(j_start, end):
            yield chr(i), chr(j), (((i-ord('a'))*26) + (j-ord('a')) + 1)

        j_start = ord('a')

def parse_file(path : str, pattern: re.Pattern) -> dict:   
    word_dict = {}
    
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern.match(line)
            if(match):
                word_dict[match[1]] = word_dict.get(match[1], 0) + int(match[4])
                

    return word_dict


def print_elapsed_time(start_time:float) -> None:
    t = time.time() - start_time

    hours   = t/(60*60)
    minutes = (hours-int(hours))*60
    seconds = (minutes-int(minutes))*60


    print("Elapsed Time - %02d:%02d:%02d" %(hours, minutes, seconds))

    
def show_download_progress(block_count: int, block_size: int, file_size: int):
    progress = int(30*(block_count*block_size)/file_size)
    progress_str = "[" + "█"*progress + "░"*(30-progress) + "]"
    print(f"\r\t{progress_str}\t{block_count*block_size:,}/{file_size:,} MB (%{100*(block_count*block_size)/file_size:2.2f})", end="")


def main(start_point) -> None:
    pattern: re.Pattern = re.compile(VALIDATION_REGEX)
    
    start_time:float = time.time()

    for a, b, i in iterate_sections(start_point):
        
        gz_path   : str = "%s.gz"   % (a+b)
        txt_path1 : str = "%s1.txt" % (a+b)
        txt_path  : str = "%s.txt"  % (a+b)


        print("Downloading section \"%s\"" % (a+b))

        try:
            x = request.urlretrieve("http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-5gram-20120701-" + (a+b) + ".gz"
                , gz_path,
                show_download_progress)
            print()

        except error.URLError as e:
            print(e)
            print("\tSection \"%s\" does not exist, continuing to the next...\n" % (a+b))
            continue

        print("\tDecompressing..")
        
        with gzip.open(gz_path, "rb") as f_gz:
            with open(txt_path1, "wb") as f_txt:
                shutil.copyfileobj(f_gz, f_txt)
        
        os.remove(gz_path)


        print("\tParsing...")

        word_dict: dict = parse_file(txt_path1, pattern)
        os.remove(txt_path1)

        print("\tSaving...")

        with open(txt_path, "w") as file:
            for k, v in word_dict.items():
                file.write("%s, %s\n" % (k, v))

        print("\tDone downloading section \"%s\"" % (a+b))
        print("%d/%d (%s%0.2f) of sections downloaded" % (i, 26*26, "%", 100*i/(26*26)) )
        print_elapsed_time(start_time)
        print("\n")



if __name__ == "__main__":
    start_point = None
    
    if(len(sys.argv)>1):
        start_point = sys.argv[1]
        if(not re.fullmatch("[a-z]{2}", start_point)):
            print("please specify the start point as two lowercase letters.")
            exit()

    main(start_point) 