from py3dbp import Packer, Bin, Item, Painter
import time
start = time.time()

'''

If you have multiple boxes, you can change distribute_items to achieve different packaging purposes.
1. distribute_items=True , put the items into the box in order, if the box is full, the remaining items will continue to be loaded into the next box until all the boxes are full  or all the items are packed.
2. distribute_items=False, compare the packaging of all boxes, that is to say, each box packs all items, not the remaining items.

'''

# init packing function
packer = Packer()
#  init bin
box = Bin(partno='example7-Bin1', width=5, height=5,
          depth=5, max_weight=100, corner=0, bin_type=1)
box2 = Bin(partno='example7-Bin2', width=3, height=3,
           depth=5, max_weight=100, corner=0, bin_type=1)
#  add item
# Item('item partno', (W,H,D), Weight, Packing Priority level, load bear, Upside down or not , 'item color')
packer.add_bin(box)
packer.add_bin(box2)

packer.add_item(Item(partno='Box-1', name='test1', typeof='cube', width=5, height=4,
                depth=1, weight=1, level=1, loadbear=100, upside_down_=True, color='yellow'))
packer.add_item(Item(partno='Box-2', name='test2', typeof='cube', width=1, height=2,
                depth=4, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-3', name='test3', typeof='cube', width=1, height=2,
                depth=3, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-4', name='test4', typeof='cube', width=1, height=2,
                depth=2, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-5', name='test5', typeof='cube', width=1, height=2,
                depth=3, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-6', name='test6', typeof='cube', width=1, height=2,
                depth=4, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-7', name='test7', typeof='cube', width=1, height=2,
                depth=2, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-8', name='test8', typeof='cube', width=1, height=2,
                depth=3, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-9', name='test9', typeof='cube', width=1, height=2,
                depth=4, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-10', name='test10', typeof='cube', width=1, height=2,
                depth=3, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-11', name='test11', typeof='cube', width=1, height=2,
                depth=2, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-12', name='test12', typeof='cube', width=5, height=4,
                depth=1, weight=1, level=1, loadbear=100, upside_down_=True, color='pink'))
packer.add_item(Item(partno='Box-13', name='test13', typeof='cube', width=1, height=1,
                depth=4, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-14', name='test14', typeof='cube', width=1, height=2,
                depth=1, weight=1, level=1, loadbear=100, upside_down_=True, color='pink'))
packer.add_item(Item(partno='Box-15', name='test15', typeof='cube', width=1, height=2,
                depth=1, weight=1, level=1, loadbear=100, upside_down_=True, color='pink'))
packer.add_item(Item(partno='Box-16', name='test16', typeof='cube', width=1, height=1,
                depth=4, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-17', name='test17', typeof='cube', width=1, height=1,
                depth=4, weight=1, level=1, loadbear=100, upside_down_=True, color='olive'))
packer.add_item(Item(partno='Box-18', name='test18', typeof='cube', width=5, height=4,
                depth=2, weight=1, level=1, loadbear=100, upside_down_=True, color='brown'))

# calculate packing
packer.pack(
    # Change distribute_items=False to compare the packing situation in multiple boxes of different capacities.
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
print("***************************************************")
for idx, b in enumerate(packer.bins):
    print("**", b.string(), "**")
    print("***************************************************")
    print("FITTED ITEMS:")
    print("***************************************************")
    volume = b.width * b.height * b.depth
    volume_t = 0
    volume_f = 0
    unfitted_name = ''
    for item in b.items:
        print("partno : ", item.partno)
        print("color : ", item.color)
        print("position : ", item.position)
        print("rotation type : ", item.rotation_type)
        print("W*H*D : ", str(item.width) + ' * ' +
              str(item.height) + ' * ' + str(item.depth))
        print("volume : ", float(item.width) *
              float(item.height) * float(item.depth))
        print("weight : ", float(item.weight))
        volume_t += float(item.width) * float(item.height) * float(item.depth)
        print("***************************************************")

    print('space utilization : {}%'.format(
        round(volume_t / float(volume) * 100, 2)))
    print('residual volume : ', float(volume) - volume_t)
    print("gravity distribution : ", b.gravity)
    print("***************************************************")
    # draw results
    painter = Painter(b)
    fig = painter.plotBoxAndItems(
        title=b.partno, alpha=0.8, write_num=False, fontsize=10
    )

print("***************************************************")
print("UNFITTED ITEMS:")
volume_f = 0
unfitted_name = ""
for item in packer.unfit_items:
    print("***************************************************")
    print('name : ', item.name)
    print("partno : ", item.partno)
    print("color : ", item.color)
    print("W*H*D : ", str(item.width) + ' * ' +
          str(item.height) + ' * ' + str(item.depth))
    print("volume : ", float(item.width) *
          float(item.height) * float(item.depth))
    print("weight : ", float(item.weight))
    volume_f += float(item.width) * float(item.height) * float(item.depth)
    unfitted_name += '{},'.format(item.partno)
    print("***************************************************")
print("***************************************************")
print('unpack item : ', unfitted_name)
print('unpack item volume : ', volume_f)

stop = time.time()
print('used time : ', stop - start)

fig.show()  # type: ignore
