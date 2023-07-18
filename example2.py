from py3dbp import Packer, Bin, Item, Painter
import time
start = time.time()

'''

This case is used to demonstrate an example of a packing complex situation.

'''

# init packing function
packer = Packer()
#  init bin
box = Bin(partno='example2', width=30, height=10, depth=15, max_weight=99, corner=0, bin_type=1)
packer.add_bin(box)
#  add item
packer.add_item(Item(partno='test1', name='test', typeof='cube', width=9, height=8, depth=7, weight=1, level=1, loadbear=100, upside_down_=True, color='red'))
packer.add_item(Item(partno='test2', name='test', typeof='cube', width=4, height=25, depth=1, weight=1, level=1, loadbear=100, upside_down_=True, color='blue'))
packer.add_item(Item(partno='test3', name='test', typeof='cube', width=2, height=13, depth=5, weight=1, level=1, loadbear=100, upside_down_=True, color='gray'))
packer.add_item(Item(partno='test4', name='test', typeof='cube', width=7, height=5, depth=4, weight=1, level=1, loadbear=100, upside_down_=True, color='orange'))
packer.add_item(Item(partno='test5', name='test', typeof='cube', width=10, height=5, depth=2, weight=1, level=1, loadbear=100, upside_down_=True, color='lawngreen'))
packer.add_item(Item(partno='test6', name='test', typeof='cube', width=6, height=5, depth=2, weight=1, level=1, loadbear=100, upside_down_=True, color='purple'))
packer.add_item(Item(partno='test7', name='test', typeof='cube', width=5, height=2, depth=9, weight=1, level=1, loadbear=100, upside_down_=True, color='yellow'))
packer.add_item(Item(partno='test8', name='test', typeof='cube', width=10, height=8, depth=5, weight=1, level=1, loadbear=100, upside_down_=True, color='pink'))
packer.add_item(Item(partno='test9', name='test', typeof='cube', width=1, height=3, depth=5, weight=1, level=1, loadbear=100, upside_down_=True, color='brown'))
packer.add_item(Item(partno='test10', name='test', typeof='cube', width=8, height=4, depth=7, weight=1, level=1, loadbear=100, upside_down_=True, color='cyan'))
packer.add_item(Item(partno='test11', name='test', typeof='cube', width=2, height=5, depth=3, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='test12', name='test', typeof='cube', width=1, height=9, depth=2, weight=1, level=1, loadbear=100, upside_down_=True, color='darkgreen'))
packer.add_item(Item(partno='test13', name='test', typeof='cube', width=7, height=5, depth=4, weight=1, level=1, loadbear=100, upside_down_=True, color='orange'))
packer.add_item(Item(partno='test14', name='test', typeof='cube', width=10, height=2, depth=1, weight=1, level=1, loadbear=100, upside_down_=True, color='lawngreen'))
packer.add_item(Item(partno='test15', name='test', typeof='cube', width=3, height=2, depth=4, weight=1, level=1, loadbear=100, upside_down_=True, color='purple'))
packer.add_item(Item(partno='test16', name='test', typeof='cube', width=5, height=7, depth=8, weight=1, level=1, loadbear=100, upside_down_=True, color='yellow'))
packer.add_item(Item(partno='test17', name='test', typeof='cube', width=4, height=8, depth=3, weight=1, level=1, loadbear=100, upside_down_=True, color='white'))
packer.add_item(Item(partno='test18', name='test', typeof='cube', width=2, height=11, depth=5, weight=1, level=1, loadbear=100, upside_down_=True, color='brown'))
packer.add_item(Item(partno='test19', name='test', typeof='cube', width=8, height=3, depth=5, weight=1, level=1, loadbear=100, upside_down_=True, color='cyan'))
packer.add_item(Item(partno='test20', name='test', typeof='cube', width=7, height=4, depth=5, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='test21', name='test', typeof='cube', width=2, height=4, depth=11, weight=1, level=1, loadbear=100, upside_down_=True, color='darkgreen'))
packer.add_item(Item(partno='test22', name='test', typeof='cube', width=1, height=3, depth=4, weight=1, level=1, loadbear=100, upside_down_=True, color='orange'))
packer.add_item(Item(partno='test23', name='test', typeof='cube', width=10, height=5, depth=2, weight=1, level=1, loadbear=100, upside_down_=True, color='lawngreen'))
packer.add_item(Item(partno='test24', name='test', typeof='cube', width=7, height=4, depth=5, weight=1, level=1, loadbear=100, upside_down_=True, color='purple'))
packer.add_item(Item(partno='test25', name='test', typeof='cube', width=2, height=10, depth=3, weight=1, level=1, loadbear=100, upside_down_=True, color='yellow'))
packer.add_item(Item(partno='test26', name='test', typeof='cube', width=3, height=8, depth=1, weight=1, level=1, loadbear=100, upside_down_=True, color='pink'))
packer.add_item(Item(partno='test27', name='test', typeof='cube', width=7, height=2, depth=5, weight=1, level=1, loadbear=100, upside_down_=True, color='brown'))
packer.add_item(Item(partno='test28', name='test', typeof='cube', width=8, height=9, depth=5, weight=1, level=1, loadbear=100, upside_down_=True, color='cyan'))
packer.add_item(Item(partno='test29', name='test', typeof='cube', width=4, height=5, depth=10, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='test30', name='test', typeof='cube', width=10, height=10, depth=2, weight=1, level=1, loadbear=100, upside_down_=True, color='darkgreen'))

# calculate packing
packer.pack(
    bigger_first=True,
    distribute_items=True,
    fix_point=True,
    check_stable=True,
    support_surface_ratio=0.75,
    number_of_decimals=0
)

# print result
b = packer.bins[0]
volume = b.width * b.height * b.depth
print(":::::::::::", b.string())

print("FITTED ITEMS:")
volume_t = 0
volume_f = 0
unfitted_name = ''
for item in b.items:
    print("partno : ", item.partno)
    print("color : ", item.color)
    print("position : ", item.position)
    print("rotation type : ", item.rotation_type)
    print("W*H*D : ", str(item.width) + '*' +
          str(item.height) + '*' + str(item.depth))
    print("volume : ", float(item.width) *
          float(item.height) * float(item.depth))
    print("weight : ", float(item.weight))
    volume_t += float(item.width) * float(item.height) * float(item.depth)
    print("***************************************************")
print("***************************************************")
print("UNFITTED ITEMS:")
for item in b.unfitted_items:
    print("partno : ", item.partno)
    print("color : ", item.color)
    print("W*H*D : ", str(item.width) + '*' +
          str(item.height) + '*' + str(item.depth))
    print("volume : ", float(item.width) *
          float(item.height) * float(item.depth))
    print("weight : ", float(item.weight))
    volume_f += float(item.width) * float(item.height) * float(item.depth)
    unfitted_name += '{},'.format(item.partno)
    print("***************************************************")
print("***************************************************")
print('space utilization : {}%'.format(
    round(volume_t / float(volume) * 100, 2)))
print('residual volume : ', float(volume) - volume_t)
print('unpack item : ', unfitted_name)
print('unpack item volume : ', volume_f)
print("gravity distribution : ", b.gravity)
stop = time.time()
print('used time : ', stop - start)

# draw results
painter = Painter(b)
fig = painter.plotBoxAndItems(
    title=b.partno,
    alpha=0.8,
    write_num=False,
    fontsize=10
)
fig.show()  # type: ignore
