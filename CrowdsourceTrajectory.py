import numpy as np
import random
import pandas as pd
import os
from collections import OrderedDict

# from UtilityFunctions import write_list_to_file

'''
This class is used to generate triples whose values will be crowd-sourced. The most important inputs will be a list of 
objects, a list of subjects, and a relation between them. Gold standard pairs known to be true should also be provided
to filter workers' inputs.
The basic usage of this class is demonstrated below:
    # Step 1. Initialize class
    from CrowdsourceKitchenConstants import objects, location_on, location_on_gold
    CTV = CrowdsourcingTripleValues(objects, location_on, "on", location_on_gold, "./data/crowdsource/on")
    
    # Step 2. Generate all possible triple combinations of subjects and objects. This function should only be called once.
    CTV.generate_queries()
    
    # Step 3. Parse and filter workers' inputs and generate additional triples. When a user fails to answer the gold
    #         standard correctly, a task including multiple queries and the gold standard will be reassigned to another
    #         worker in the additional queries. When all gold standards have been answered correctly, this function will
    #         then check if workers answering the same query are agreeing to each other. Additional queries will be 
    #         generated for disagreed queries. This function should be called multiple times until all inputs are
    #         correct according to gold standards and all queries have agreed answers. 
    CTV.parse_results_and_generate_additional_queries()

    example:
    # 2. "on" relation: Finished
    # from CrowdsourceKitchenConstants import objects, location_on, location_on_gold
    # CTV = CrowdsourcingTripleValues(objects, location_on, "on", location_on_gold, "./data/crowdsource/on")
    # # CTV.generate_queries()
    # CTV.parse_results_and_generate_additional_queries()
    # CTV.print_statistics()
    # CTV.parse_results_and_generate_additional_queries()
    # CTV.print_statistics()
    # CTV.parse_results_and_generate_additional_queries()
    # CTV.print_statistics()
    # CTV.parse_results_and_generate_additional_queries()
    # CTV.print_statistics()
'''

# The assumption is that for a relation, there is a set of subjects and objects.
class CrowdsourcingTripleValues:

    def __init__(self, videos, gold_standards, directory, queries_per_hit=10, workers_per_hit=5, gold_standard_position=7):
        self.videos = videos
        self.gold_standards = gold_standards
        self.gold_standard_position = gold_standard_position
        if not os.path.isdir(directory):
            os.mkdir(directory)
        self.directory = directory
        self.queries_per_hit = queries_per_hit
        self.workers_per_hit = workers_per_hit

        # Used to count the number of times used to collect all the data due to answers that violate gold standards
        self.round = 1

        # This stores counts of different responces for each video
        self.result = OrderedDict()
        # This stores final evaluation of each video
        self.data = OrderedDict()
        for video in self.videos:
            # list values correspond to numbers of True, False, Unclear
            self.result[video] = [0,0,0]
            self.data[video] = None

    def generate_queries(self):
        print("Generate the initial set of queries...")
        if self.gold_standard_position > self.queries_per_hit:
            print("gold standard position out of query size and can't be last one")
            exit()

        filename = os.path.join(self.directory, "query_round" + str(self.round) + ".csv")
        with open(filename, "w+") as fh:
            # 1. generate headers
            for i in range(self.queries_per_hit):
                # below is the header format
                fh.write("video" + str(i + 1))
                if i + 1 < self.queries_per_hit:
                    fh.write(",")
                else:
                    fh.write("\n")

            # 2. generate all queries
            # make query number dividable by queries_per_hit - 1 because MTurk requires all rows have the same number of
            # columns for the input file
            add_number = (self.queries_per_hit - (len(self.videos) % (self.queries_per_hit - 1))) % self.queries_per_hit - 1
            add_pos = np.random.choice(len(self.videos), add_number)
            for i in add_pos:
                self.videos.append(self.videos[i])
            # shuffle
            random.shuffle(self.videos)

            print("Writing " + str(len(self.videos)/(self.queries_per_hit-1)) + " hits...")
            # 3. write all queries to file
            counter = 0
            for q in self.videos:
                if counter + 1 == self.gold_standard_position:
                    fh.write(self.gold_standards[np.random.randint(len(self.gold_standards))] + ", ")
                    counter += 1
                fh.write(q)
                if counter + 1 < self.queries_per_hit:
                    counter += 1
                    fh.write(",")
                else:
                    fh.write("\n")
                    counter = 0

    def parse_results_and_generate_additional_queries(self, overwrite=False):
        print("\nRound " + str(self.round) + ": Parsing crowdsourcing results and generate additional queries if necessary...")

        # 1. Read CSV and store result
        results_filename = os.path.join(self.directory, "result_round" + str(self.round) + ".csv")
        print(results_filename)
        df = pd.read_csv(results_filename)

        # 2. Filter results according to gold standards
        reject_ids = []
        for i in range(df.shape[0]):

            # Check if gold question is answered correctly
            header_ans = "Answer.value" + str(self.gold_standard_position)
            header_rel = "Input.relation" + str(self.gold_standard_position)
            video = df.ix[i, header_rel]
            # IMPORTANT: the logic for rejecting answers based on gold standards is that if an entity pair is specified
            # in gold standard and the answer has value 'false', OR an entity pair is not specified in gold standard and the
            # answer has value 'true'
            if video in self.gold_standards and df.ix[i, header_ans] == "False":
                #or ((subj.split(".")[0].lower(), obj.split(".")[0].lower()) not in self.gold_standards and df.ix[i, header_ans] == "True"):
                reject_ids.append(i)
                header_rel = "Input.relation" + str(self.gold_standard_position)
                print("Reject " + str(i) + " because: " + df.ix[i, header_rel] + "---" + df.ix[i, header_ans])
                # ToDO: write to the results_filename to mark accept and reject
            # if df.ix[i, header_ans] == "False":
            #     # DEBUG: Four lines below is used to ignore falsely defined gold standard
            #     header_rel = "Input.relation" + str(self.gold_standard_position)
            #     subj, obj = self.get_entity_pair(df.ix[i, header_rel])
            #     if subj == 'cutting_board.n.01' and obj == 'kitchen_stove.n.01':
            #         continue
            #
            #     reject_ids.append(i)
            #     header_rel = "Input.relation" + str(self.gold_standard_position)
            #     print("Reject " + str(i) + " because: " + df.ix[i, header_rel] + "---" + df.ix[i, header_ans])
            #     # ToDO: write to the results_filename to mark accept and reject
            #
            else:
                for qi in range(self.queries_per_hit):
                    if qi + 1 == self.gold_standard_position:
                        continue
                    header_rel = "Input.relation" + str(qi+1)
                    header_ans = "Answer.value" + str(qi+1)
                    video = df.ix[i, header_rel]
                    answer = df.ix[i, header_ans]
                    if answer == "True":
                        self.result[video][0] += 1
                    elif answer == "False":
                        self.result[video][1] += 1
                    else:
                        self.result[video][2] += 1
        print("Reject ", len(reject_ids), "hits in total.\n")

        # # 3. Process results to find disagreement
        # # Extract subject and object pairs that workers don't agree on
        # pairs_disagree = set()
        # pairs_unfinished = set()
        # for subj in self.result:
        #     for obj in self.result[subj]:
        #         # print(sum(self.result[subj][obj]))
        #         if sum(self.result[subj][obj]) < self.workers_per_hit:
        #             pairs_unfinished.add(tuple([subj, obj]))
        #             continue
        #         if np.max(self.result[subj][obj]) < (len(self.result[subj][obj]) / 2 + 1):
        #             # print("crowdsourced result for (" + subj + ", " + self.relation + ", " + obj + ") workers do not agree on")
        #             pairs_disagree.add(tuple([subj, obj]))
        # for subj, obj in pairs_disagree:
        #     print("disagreed (" + subj + ", " + self.relation + ", " + obj + "): " + str(self.result[subj][obj]))
        # print("disagreed pairs:", len(pairs_disagree), "\n")

        # for subj, obj in pairs_unfinished:
        #     print(
        #         "unfinished (" + subj + ", " + self.relation + ", " + obj + "): " + str(
        #             self.result[subj][obj]))
        # print("unfinished pairs:", len(pairs_unfinished), "\n")

        # 4. Write new queries for rejected or disagreed results
        if len(reject_ids) > 0: #or len(pairs_disagree) > 0:
            self.round += 1
            if overwrite:
                queries_filename = os.path.join(self.directory, "query_" + self.relation + "_round" + str(self.round) + ".csv")
                # Generate additional queries filtered out by gold standard
                print("Generating additional queries...")
                additional_queries = open(queries_filename, "w")

                # (1). Generate headers
                for i in range(self.queries_per_hit):
                    additional_queries.write("relation" + str(i + 1))
                    if i + 1 < self.queries_per_hit:
                        additional_queries.write(",")
                    else:
                        additional_queries.write("\n")

                # IMPORTANT: only post queries based on disagreement when initial set of data are complete which means no data have been rejected.
                if len(reject_ids) > 0:
                    # A. For rejected
                    # (2)A. write queries
                    print("Writing " + str(len(reject_ids)) + " hits for rejected...")
                    for i in reject_ids:
                        for qi in range(self.queries_per_hit):
                            header_rel = "Input.relation" + str(qi + 1)
                            additional_queries.write(df.ix[i, header_rel])
                            if qi + 1 < self.queries_per_hit:
                                additional_queries.write(",")
                            else:
                                additional_queries.write("\n")
                # else:
                #     # B. For disagreed
                #     # (2)B. generate all queries
                #     all_questions = []
                #     for pair in pairs_disagree:
                #         all_questions.append(self.get_str(pair))
                #     # make query number dividable by queries_per_hit - 1 because MTurk requires all rows have the same number of
                #     # columns for the input file
                #     add_number = ((self.queries_per_hit - 1) - (
                #                 len(all_questions) % (self.queries_per_hit - 1)))
                #     add_pos = np.random.choice(len(all_questions), add_number)
                #     for i in add_pos:
                #         all_questions.append(all_questions[i])
                #     # shuffle
                #     all_questions += all_questions
                #     random.shuffle(all_questions)

                #     print("\nWriting 2 * " + str(len(all_questions) / (self.queries_per_hit - 1) / 2)  + " hits for disagreed...")
                #     print("^Writing hits 2 times in order to have 2 workers per hit")
                #     # (3)B. write all queries to file
                #     counter = 0
                #     for q in all_questions:
                #         if counter + 1 == self.gold_standard_position:
                #             additional_queries.write(self.get_str(
                #                 self.gold_standards[np.random.randint(len(self.gold_standards))]) + ", ")
                #             counter += 1
                #         additional_queries.write(q)
                #         if counter + 1 < self.queries_per_hit:
                #             counter += 1
                #             additional_queries.write(",")
                #         else:
                #             additional_queries.write("\n")
                #             counter = 0
                #     print("*****Set 1 worker per hit on MTurk*****")

    def process_results(self):
        print("\nProcessing results of all rounds and write to file...")
        # Check is raw results is complete

        number_complete = 0.0
        for video in self.result:
            if sum(self.result[video]) < self.workers_per_hit:
                print("crowdsourced result for (" + subj + ", " + self.relation + ", " + obj + ") not complete")
            # elif np.max(self.result[subj][obj]) < (len(self.result[subj][obj]) / 2 + 1):
            #     print("crowdsourced result for (" + subj + ", " + self.relation + ", " + obj + ") workers do not agree on")
            else:
                number_complete += 1
                # previous elif already ensures only one max exists
                bool_idx = np.argmax(self.result[video])
                if bool_idx == 0:
                    self.data[video] = True
                elif bool_idx == 1:
                    self.data[video] = False
                else:
                    self.data[video] = None

        # print data statistics
        print("\nData Statistics:")
        print("Completion rate:", number_complete / len(self.videos))

    def print_statistics(self):
        # Check is raw results is complete
        trues = []
        falses = []
        unknowns = []
        number_complete = 0.0
        for subj in self.result:
            for obj in self.result[subj]:
                if sum(self.result[subj][obj]) < self.workers_per_hit:
                    pass
                    # print("crowdsourced result for (" + subj + ", " + self.relation + ", " + obj + ") not complete")
                else:
                    bool_idx = np.argmax(self.result[subj][obj])
                    if np.max(self.result[subj][obj]) < 3:
                        # print("crowdsourced result for (" + subj + ", " + self.relation + ", " + obj + ") workers do not agree on")
                        continue
                    number_complete += 1
                    if bool_idx == 0:
                        self.data[subj][obj] = True
                        trues.append((subj, self.relation, obj))
                    elif bool_idx == 1:
                        self.data[subj][obj] = False
                        falses.append((subj, self.relation, obj))
                    else:
                        self.data[subj][obj] = None
                        unknowns.append((subj, self.relation, obj))

        # print data statistics
        print("\nData Statistics:")
        print("Completion rate:", number_complete / len(self.subjects) / len(self.objects))

if __name__ == "__main__":

    videos = []
    with open('video_urls.txt', "r") as video_file:
        for line in video_file:
            if line == "\n":
                continue
            videos.append(line.replace("\n", ""))

    gold_standards = []
    with open('gold_urls.txt', "r") as gold_file:
        for line in gold_file:
            if line == "\n":
                continue
            gold_standards.append(line.replace("\n", ""))

    print videos


    CTV = CrowdsourcingTripleValues(videos, gold_standards, "./data", 5, 5, 3)
    CTV.generate_queries()

    # CTV.parse_results_and_generate_additional_queries()
    # CTV.print_statistics()
    
    # CTV.parse_results_and_generate_additional_queries(overwrite=True)
    # CTV.print_statistics()
    # CTV.parse_results_and_generate_additional_queries()
    # CTV.print_statistics()
    # CTV.parse_results_and_generate_additional_queries()
    # CTV.print_statistics()