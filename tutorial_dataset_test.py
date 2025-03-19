from tutorial_dataset import MyDataset_1, MyDataset_4, MyDataset_3

dataset1 = MyDataset_1()
dataset4 = MyDataset_4()
dataset3 = MyDataset_3()

print("dataset1:",len(dataset1))
item = dataset1[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

print(20*"=")
print("dataset4:",len(dataset4))
item = dataset4[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

print(20*"=")
print("dataset3:",len(dataset3))
item = dataset3[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

