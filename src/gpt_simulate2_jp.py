# note from sam - ive commented out/removed all occurrences of probs/logprobs, ctrl f for probs AND for change to see changes (including logprobs and others)
# ingrid - (also gpt5 doesnt support logit_bias); CHOOSING GPT4 for now despite the price (the other model mini performed too poorly, it seems)
# REMEMBER TO CHECK IF OTHER = T/F

# (do 15 per condition ABCD/EFGH for 30 per language, or only 8 for 16?)

from openai import OpenAI
import pathlib
import pandas as pd
import numpy as np
import datetime
import time
import argparse

def input_parse():
    parser = argparse.ArgumentParser(description='Run the GPT simulation')

    parser.add_argument("-other", "--other_letters", help='If specified, use other deck names (E, F, G, H)', action="store_true")
    parser.add_argument("-nudge", "--nudge_choices", help='Whether to use nudging parameters', action="store_true") #this goes unused; only there if "less ideal" (noisier!) decision-making is desired from the model

    return parser.parse_args()

def create_deck_dict(payoff_df, deck_name): 
    '''
    Create a deck dictionary for each deck 
    '''
    deck_dict = {}

    for i, row in payoff_df.iterrows(): 
        deck_dict[row["card_number"]] = {f"win": row[f"win_{deck_name}"], f"loss": row[f"loss_{deck_name}"]}

    return deck_dict

def create_all_decks(other=False):
    '''
    Create all decks from the payoff scheme as seperate dicts
    '''
    path = pathlib.Path(__file__)
    filepath = path.parents[1] / "utils" / "payoff_scheme_3.csv"

    df = pd.read_csv(filepath)

    decks = {}

    for deck_name in ["A", "B", "C", "D"]:
        decks[deck_name] = create_deck_dict(df, deck_name)
    
    if other: 
        decks["E"] = decks.pop("A")
        decks["F"] = decks.pop("B")
        decks["G"] = decks.pop("C")
        decks["H"] = decks.pop("D")

    return decks

def create_task_description(task_desc_path, other=False):
    with open(task_desc_path / 'modified_task_desc.txt', 'r') as f:
        task_desc = f.read() 
    
    if other:
        task_desc = task_desc.replace("A、B、C、D", "E、F、G、H")

    return task_desc

def initialize_client():
    '''
    Intialize the OpenAI class
    '''
    with open(pathlib.Path(__file__).parents[1] / "keys" / 'api_key.txt', 'r') as f:
        key = f.read()

    with open(pathlib.Path(__file__).parents[1] / "keys" / 'org_id.txt', 'r') as f:
        org_id = f.read()

    client = OpenAI(
        organization=org_id,
        api_key=key
    )

    return client

def select_deck(client, task_desc, model_endpoint = 'gpt-4.1', updated_messages=None, other=False, empty_deck=None, frequency_penalty=0, presence_penalty=0): 
    '''
    Select a deck from the four decks with ChatGPT
    '''
    if other:
        letters = ["E", "F", "G", "H"]
        logit_bias = {36: 100, 37: 100, 38: 100, 39: 100}
    else:
        letters = ["A", "B", "C", "D"]
        logit_bias = {32: 100, 33: 100, 34: 100, 35: 100}
    
    d1 = letters[0]
    d2 = letters[1]
    d3 = letters[2]
    d4 = letters[3]

    if updated_messages is None:
        messages = [
            {"role": "system", "content": task_desc},
            {"role": "user", "content": task_desc + "\n" + f"'{d1}、{d2}、{d3}、それとも{d4}？'"}
        ]
    else: 
        messages = updated_messages
    
    if empty_deck:
        letters.remove(empty_deck)

    valid_deck = False
    attempt_count = 0
    max_attempts = 3 

    while not valid_deck and attempt_count < max_attempts:
        completion = client.chat.completions.create(
            model=model_endpoint,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=1,
            messages=messages,
            max_completion_tokens=1,
            logit_bias=logit_bias
            )

        card_selection = completion.choices[0].message.content

        if card_selection in letters:
            valid_deck = True
        else:
            messages.append({"role": "assistant", "content": card_selection})
            available_decks = ", ".join(letters)
            messages.append({"role": "user", "content": f"'使用したいデッキのアルファベットのみを指定してください。利用可能なデッキは {available_decks} です。'"})
            attempt_count += 1
            time.sleep(5)

    if not valid_deck:
        print(messages)
        raise ValueError("Maximum attempts reached, ChatGPT failed to select a valid deck.")

    return card_selection, messages


def get_payoff(card_selection, decks, selected):
    '''
    Get the payoff for the selected card
    '''
    deck = decks[card_selection]

    selected[card_selection] += 1

    card_number = selected[card_selection]

    win = deck[card_number]["win"]       
    loss = deck[card_number]["loss"]

    return selected, win, loss

def check_for_empty_deck(selected):
    '''
    Checks if any of the four decks have been selected 60 times
    '''
    for deck in selected:
        if selected[deck] >= 60:
            return deck
    
    return None

def update_messages(messages, card_selection, win, loss, total_earnings, empty_deck, other=False):
    '''
    Update ChatGPT messages with card selection and payoff. Add empty deck message if needed.
    '''
    if other:
        letters = ["E", "F", "G", "H"]
    else:
        letters = ["A", "B", "C", "D"]
    
    d1 = letters[0]
    d2 = letters[1]
    d3 = letters[2]
    d4 = letters[3]

    messages.append({"role": "assistant", "content": card_selection})

    payoff_message = f" あなたは${win}ドル勝ちました！"
    
    if loss > 0:
        loss_message = f"しかし、${loss}ドル負けました！"
        payoff_message += "\n" + loss_message

    payoff_message += f"\n総獲得額（TOTAL EARNINGS）：${total_earnings}。"

    payoff_message += f"\n借入金：$2000。"

    messages.append({"role": "user", "content": payoff_message})

    if empty_deck is None:
        messages.append({"role": "user", "content": f"'{d1}、{d2}、{d3}、それとも{d4}？'"})

    else:
        not_empty = [d for d in letters if d != empty_deck]

        new_message = f"デッキ{empty_deck}は空です。{not_empty[0]}、{not_empty[1]}、それとも{not_empty[2]}？"

        messages.append({"role": "user", "content": new_message})

    return messages

def save_data(data:dict, updated_messages, data_path, other=False, nudged = False):
    '''
    Save data as a csv and save messages as a txt
    '''
    data_path.mkdir(parents=True, exist_ok=True)

    log_path = data_path / "message_logs"
    log_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame.from_dict(data, orient="index")

    df.index.name = "trial"
    df.reset_index(inplace=True)
    
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")

    if other:
        filename = f"gpt_{formatted_time}_EFGH"
    else:
        filename = f"gpt_{formatted_time}_ABCD"

    if nudged:
        filename = filename + "_nudged"
    
    df.to_csv(data_path/ f"{filename}.csv", index=False)

    with open(log_path / f"{filename}.txt", 'w') as f:
        f.write(str(updated_messages))

        
def main():
    args = input_parse()

    path = pathlib.Path(__file__).parents[1]
    task_desc_path = path / "utils" / "task_descriptions"
    
    if args.other_letters:
        letters = ["E", "F", "G", "H"]
    else:
        letters = ["A", "B", "C", "D"]
        
    if args.nudge_choices:
        presence_penalty = 1
        frequency_penalty = 2
    else:
        presence_penalty = 0
        frequency_penalty = 0

    decks = create_all_decks(other=args.other_letters) #if true or false
    d1, d2, d3, d4 = letters[0], letters[1], letters[2], letters[3]

    client = initialize_client()

    task_desc = create_task_description(task_desc_path, other=args.other_letters)

    trials_to_play = 100
    trials_played = 0
    selected = {d1:0, d2:0, d3:0, d4:0}
    total_earnings = 2000
    
    data = {}

    updated_messages = None
    empty_deck = None

    secs = 10

    while trials_played < trials_to_play:
        time.sleep(secs)

        card_selection, messages= select_deck(client, task_desc, updated_messages=updated_messages, other=args.other_letters, empty_deck=empty_deck, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

        selected, win, loss = get_payoff(card_selection, decks, selected)

        empty_deck = check_for_empty_deck(selected)

        total_earnings = total_earnings + win - loss

        trials_played += 1

        data[trials_played] = {"deck": card_selection, "gain": win, "loss": loss}

        updated_messages = update_messages(messages, card_selection, win, loss, total_earnings, empty_deck, other=args.other_letters)

        print(updated_messages[-4:-1])
        print(trials_played, card_selection, win, loss)
        print()

    data_path = path / "data" / "raw_data" / "GPTdata"
    save_data(data, updated_messages, data_path=data_path, other=args.other_letters, nudged = args.nudge_choices)

if __name__ == '__main__':
    main()