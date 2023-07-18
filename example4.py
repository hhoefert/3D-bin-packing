from py3dbp import Packer, Bin, Item, Painter
import time
start = time.time()

'''

This example can be used to test large batch calculation time and binding functions.

'''

# init packing function
packer = Packer()

# Evergreen Real Container (20ft Steel Dry Cargo Container)
# Unit cm/kg
box = Bin(
    partno='example4', width=589.8, height=243.8, depth=259.1, max_weight=28080, corner=15, bin_type=1
)

packer.add_bin(box)

# dyson DC34 (20.5 * 11.5 * 32.2 ,1.33kg)
# 64 pcs per case ,  82 * 46 * 170 (85.12)
for i in range(15):
    packer.add_item(Item(
        partno='Dyson DC34 Animal{}'.format(str(i+1)), name='Dyson', typeof='cube', width=170, height=82, depth=46, weight=85.12, level=1, loadbear=100, upside_down_=True, color='#FF0000')
    )

# washing machine (85 * 60 *60 ,10 kG)
# 1 pcs per case, 85 * 60 *60 (10)
for i in range(18):
    packer.add_item(Item(
        partno='wash{}'.format(str(i+1)), name='wash', typeof='cube', width=85, height=60, depth=60, weight=10, level=1, loadbear=100, upside_down_=True, color='#FFFF37'
    ))

# 42U standard cabinet (60 * 80 * 200 , 80 kg)
# one per box, 60 * 80 * 200 (80)
for i in range(15):
    packer.add_item(Item(
        partno='Cabinet{}'.format(str(i+1)), name='cabinet', typeof='cube', width=60, height=80, depth=200, weight=80, level=1, loadbear=100, upside_down_=True, color='#842B00')
    )

# Server (70 * 100 * 30 , 20 kg)
# one per box , 70 * 100 * 30 (20)
for i in range(42):
    packer.add_item(Item(
        partno='Server{}'.format(str(i+1)), name='server', typeof='cube', width=70, height=100, depth=30, weight=20, level=1, loadbear=100, upside_down_=True, color='#0000E3')
    )


# calculate packing
packer.pack(
    # binding=[('server','cabinet','wash')], # binding=['cabinet','wash','server'], number_of_decimals=0
    bigger_first=True, distribute_items=False, fix_point=True, check_stable=True, support_surface_ratio=0.75,
)

# print result
for box in packer.bins:

    volume = box.width * box.height * box.depth
    print(":::::::::::", box.string())

    print("FITTED ITEMS:")
    volume_t = 0
    volume_f = 0
    unfitted_name = ''

    # '''
    for item in box.items:
        print("partno : ", item.partno)
        print("type : ", item.name)
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
    # '''
    print("UNFITTED ITEMS:")
    for item in box.unfitted_items:
        print("partno : ", item.partno)
        print("type : ", item.name)
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
    print("gravity distribution : ", box.gravity)
    # '''
    stop = time.time()
    print('used time : ', stop - start)

    # draw results
    painter = Painter(box)
    fig = painter.plotBoxAndItems(
        title=box.partno, alpha=0.2, write_num=False, fontsize=6
    )
fig.show()  # type: ignore
