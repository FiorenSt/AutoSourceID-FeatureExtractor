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

def main():
    # ---------------------
    # Train the Part I model
    # ---------------------

    print("Training Part I Model...")
    train_dataset, valid_dataset, total_train, total_val = prepare_data(config)

    # Train the model
    train_model_part1(train_dataset, valid_dataset, total_train, total_val)

    # ---------------------
    # Train the Part II model
    # ---------------------
    print("Training Part II Model...")

    # Train the model
    train_model_part2(train_dataset, valid_dataset, total_train, total_val, config)

if __name__ == "__main__":
    main()