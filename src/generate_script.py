from argument import define_arguments

parser, args = define_arguments(script=True)

list_arg = set()
for arg in vars(args):
    if type(getattr(args, arg)) == list:
        if len(getattr(args, arg)) == 1:
            setattr(args, arg, getattr(args, arg)[0])
        else:
            list_arg.add(arg)

loop_cnt = 0

for arg in vars(args):
    if arg in list_arg:
        loop_cnt += 1
        attrs = getattr(args, arg)
        print("{}for {} in {}; do".format(
            "\t"*(loop_cnt-1),
            arg,
            ' '.join(list(map(str, attrs)))))

print("{}python3 train.py \\".format("\t"*loop_cnt))
for arg in vars(args):
    if arg in list_arg:
        print("{}--{} {} \\".format("\t"*loop_cnt, arg, "${{{}}}".format(arg)))
    else:
        if getattr(args, arg) != parser.get_default(arg):
            print("{}--{} {} \\".format(
                "\t"*loop_cnt, arg, getattr(args, arg)))
print()

for idx in range(loop_cnt):
    print("{}done".format("\t"*(loop_cnt-idx-1)))

