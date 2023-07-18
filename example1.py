from py3dbp import Packer, Bin, Item, Painter
import time
start = time.time()

'''

This example is used to demonstrate the mixed packing of cube and cylinder.

'''

# init packing function
packer = Packer()
#  init bin
box = Bin(partno='example1', width=5.6875, height=10.75,
          depth=15.0, max_weight=70.0, corner=0, bin_type=1)
packer.add_bin(box)
#  add item
packer.add_item(Item(partno='50g [powder 1]', name="test", typeof='cube',
                     width=2, height=2, depth=4, weight=1, level=1, loadbear=100, upside_down_=True, color='red'))
packer.add_item(Item(partno='50g [powder 2]', name="test", typeof='cube',
                     width=2, height=2, depth=4, weight=2, level=1, loadbear=100, upside_down_=True, color='blue'))
packer.add_item(Item(partno='50g [powder 3]', name="test", typeof='cube',
                     width=2, height=2, depth=4, weight=3, level=1, loadbear=100, upside_down_=True, color='gray'))
packer.add_item(Item(partno='50g [powder 4]', name="test", typeof='cube',
                     width=2, height=2, depth=4, weight=3, level=1, loadbear=100, upside_down_=True, color='orange'))
packer.add_item(Item(partno='50g [powder 5]', name="test", typeof='cylinder',
                     width=2, height=2, depth=4, weight=3, level=1, loadbear=100, upside_down_=True, color='lawngreen'))
packer.add_item(Item(partno='50g [powder 6]', name="test", typeof='cylinder', width=2,
                height=2, depth=4, weight=3, level=1, loadbear=100, upside_down_=True, color='purple'))
packer.add_item(Item(partno='50g [powder 7]', name="test", typeof='cylinder', width=1,
                height=1, depth=5, weight=3, level=1, loadbear=100, upside_down_=True, color='yellow'))
packer.add_item(Item(partno='250g [powder 8]', name="test", typeof='cylinder', width=4,
                height=4, depth=2, weight=4, level=1, loadbear=100, upside_down_=True, color='pink'))
packer.add_item(Item(partno='250g [powder 9]', name="test", typeof='cylinder', width=4,
                height=4, depth=2, weight=5, level=1, loadbear=100, upside_down_=True, color='brown'))
packer.add_item(Item(partno='250g [powder 10]', name="test", typeof='cube', width=4,
                height=4, depth=2, weight=6, level=1, loadbear=100, upside_down_=True, color='cyan'))
packer.add_item(Item(partno='250g [powder 11]', name="test", typeof='cylinder', width=4,
                height=4, depth=2, weight=7, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='250g [powder 12]', name="test", typeof='cylinder', width=4,
                height=4, depth=2, weight=8, level=1, loadbear=100, upside_down_=True, color='darkgreen'))
packer.add_item(Item(partno='250g [powder 13]', name="test", typeof='cube', width=4,
                height=4, depth=2, weight=9, level=1, loadbear=100, upside_down_=True, color='orange'))

# calculate packing
packer.pack(
    bigger_first=True,
    distribute_items=False,
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
    alpha=0.2,
    write_num=False,
    fontsize=5
)
fig.show()  # type: ignore
