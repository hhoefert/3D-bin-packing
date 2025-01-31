from py3dbp import Packer, Bin, Item, Painter
import time
start = time.time()

'''

Check stability on item - second rule
1. If the ratio below the support surface does not exceed this ratio, then check the second rule.
2. If there is no support under any of the bottom four vertices of the item, then remove the item.

'''

# init packing function
packer = Packer()
#  init bin
box = Bin(partno='example6', width=5, height=4,
          depth=7, max_weight=100, corner=0, bin_type=1)
#  add item
# Item('item partno', (W,H,D), Weight, Packing Priority level, load bear, Upside down or not , 'item color')
packer.add_bin(box)
packer.add_item(Item(partno='Box-1', name='test', typeof='cube', width=5, height=4,
                depth=1, weight=1, level=1, loadbear=100, upside_down_=True, color='yellow'))
packer.add_item(Item(partno='Box-2', name='test', typeof='cube', width=1, height=1,
                depth=4, weight=1, level=2, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-3', name='test', typeof='cube', width=3, height=4,
                depth=2, weight=1, level=3, loadbear=100, upside_down_=True, color='pink'))
packer.add_item(Item(partno='Box-4', name='test', typeof='cube', width=1, height=1,
                depth=4, weight=1, level=4, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-5', name='test', typeof='cube', width=1, height=2,
                depth=1, weight=1, level=5, loadbear=100, upside_down_=True, color='pink'))
packer.add_item(Item(partno='Box-6', name='test', typeof='cube', width=1, height=2,
                depth=1, weight=1, level=6, loadbear=100, upside_down_=True, color='pink'))
packer.add_item(Item(partno='Box-7', name='test', typeof='cube', width=1, height=1,
                depth=4, weight=1, level=7, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-8', name='test', typeof='cube', width=1, height=1, depth=4, weight=1, level=8,
                loadbear=100, upside_down_=True, color='olive'))  # Try switching WHD=(1, 1, 3) and (1, 1, 4) to compare the results
packer.add_item(Item(partno='Box-9', name='test', typeof='cube', width=5, height=4,
                depth=2, weight=1, level=9, loadbear=100, upside_down_=True, color='brown'))

# calculate packing
packer.pack(
    bigger_first=True,
    distribute_items=False,
    fix_point=True,
    check_stable=True,
    support_surface_ratio=0.75,
    number_of_decimals=0
)

# put order
packer.packing_order()

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
    title=b.partno, alpha=0.8, write_num=False, fontsize=10
)
fig.show()
