#
#                          ###           ######       ###      ### ##                ### #######    ### ######
#                         #####         ##            ###      ###  ###              ###            ###
#                        ### ###        ##            ###      ###   ###             ### #######    ###
#                       ###   ###         #####       ###      ###     ###    ###    ###            ### ######
#                      ### ### ###            ##      ###      ###   ###             ###            ###
#                     ###       ###           ##      ###      ###  ###              ###            ###
#                    ###         ###    #######       ###      ### ##                ###            ### ######
#


from model_partI import train_model as train_model_part1
from model_partII import train_model as train_model_part2  # Assuming this is the function for part2 training
from utils import load_config, prepare_data, prepare_data_part2  # Assuming prepare_data_part2 is for part 2

def main():
    # Load the configuration
    config = load_config()

    # ---------------------
    # Train the Part I model
    # ---------------------
    print("Training Part I Model...")
    train_dataset1, valid_dataset1, total_train1, total_val1 = prepare_data(config)

    # Train the model
    train_model_part1(train_dataset1, valid_dataset1, total_train1, total_val1, config)

    # ---------------------
    # Train the Part II model
    # ---------------------
    print("Training Part II Model...")
    train_dataset2, valid_dataset2, total_train2, total_val2 = prepare_data_part2(config)

    # Train the model
    train_model_part2(train_dataset2, valid_dataset2, total_train2, total_val2, config)

if __name__ == "__main__":
    main()