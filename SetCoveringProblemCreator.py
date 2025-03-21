import sys
import random
import json

class SetCoveringProblemCreator:
    def __init__(self):
        pass

    def _SampleWithoutReplacement(self, k, usize):
        return random.sample(range(1, usize + 1), k)

    def _FindMissingIntegers(self, input_set, max_num):
        all_integers_set = set(range(1, max_num + 1))
        missing_integers_set = all_integers_set - input_set
        missingIntegersList = list(missing_integers_set)
        return missingIntegersList

    def _CreateOneSet(self, usize, setOfSets, elementsCovered):
        k = random.randint(1, 10) #set size
        newSet = frozenset(self._SampleWithoutReplacement(k, usize))
        setOfSets.add(newSet)
        return elementsCovered.union(newSet)
        
    def Create(self, usize, totalSets):
        """
        The Create function generates subsets for the elements in the universe.
        usize is the total number of elements in the universe.
        totalSets is the total number of subsets that are part of the Set Covering Problem.
        The Create function returns a list of subsets as a list of lists.
        """
        if usize != 100:
            exit('Universe size (usize) must be 100.')
        setOfSets = set()
        elementsCovered = set()
        while len(setOfSets) < totalSets - 1:
            elementsCovered = self._CreateOneSet(usize, setOfSets, elementsCovered)
        missingIntegers = self._FindMissingIntegers(elementsCovered, usize)
        if len(missingIntegers) == 0:
            while len(setOfSets) < totalSets:
                elementsCovered = self._CreateOneSet(usize, setOfSets, elementsCovered)
        else:
            newSet = frozenset(missingIntegers)
            setOfSets.add(newSet)
            elementsCovered = elementsCovered.union(newSet)
        listOfSets = list(setOfSets)
        return listOfSets
    
    def WriteSetsToJson(self, listOfSets, usize, totalSets):
        # Convert frozensets to lists
        list_of_lists = [list(fs) for fs in listOfSets]
        
        # Write the list of lists to a JSON file
        fileName = f"scp_{totalSets}.json"
        with open(fileName, 'w') as json_file:
            json.dump(list_of_lists, json_file)
        
        print(f"A random instance of Set Covering Problem is created in {fileName} file:")
        print(f"universe-size = {usize}, number-of-subsets = {totalSets}.")
    
    def ReadSetsFromJson(self, fileName):
        """
        ReadSetsFromJson reads a list of lists from a json file.
        The list read will contain all the subsets in the Set Covering Problem.
        """
        try:
            with open(fileName, 'r') as json_file:
                listOfSubsets = json.load(json_file)
                
#            # Convert lists back to frozensets
#            listOfSets = [frozenset(lst) for lst in list_of_lists]
            return listOfSubsets
        except FileNotFoundError:
            print(f"Error: The file {fileName} was not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: The file {fileName} is not a valid JSON file.")
            return None

def main():
    if len(sys.argv) != 3:
        print("Error: Some command-line arguments are incorrect.")
        print("Usage: ./GenSetCoveringProblem <universe_size> <number_of_subsets>")
        print("   eg. ./EncryptTestCase 100 150")
        sys.exit()

    usize, totalSets = [int(a) for a in sys.argv[1:]]
    scp = SetCoveringProblemCreator()

    """
    The Create function in SetCoveringProblem is used as shown below.
    usize is the total number of elements in the universe.
    totalSets is the total number of subsets that are part of the Set Covering Problem.
    """
    listOfSets = scp.Create(usize, totalSets)
 
    scp.WriteSetsToJson(listOfSets, usize, totalSets)

    # Example of reading the SCP instance back from the JSON file
#    read_list_of_sets = scp.ReadSetsFromJson(f"scp_{totalSets}.json")
#    if read_list_of_sets is not None:
#        print("Read the following sets from the JSON file:")
#        for s in read_list_of_sets:
#            print(s)

if __name__ == "__main__":
    main()
