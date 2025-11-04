# def start_end(func):
#     def add_start_end(_text):
#         print('start')
#         func(_text)
#         print('end')
#     return add_start_end


# def print_text(text):
#     print(text + '!')

# start_end(print_text)('これはりんごです')

def start_end(func):
    def add_start_end(*args, **kwargs):
        print('start')
        x = func(*args, **kwargs)
        print('end')
        return x
    return add_start_end

# @start_end
# def print_join_dash(a, b):
#     print(f'{a} - {b}')

@start_end
def add_exclamation(text):
    print('add_exclamationが実行されました')
    return text + '!'

# print_join_dash('163', b='8001')
result = add_exclamation('これはりんごです')
print(result)