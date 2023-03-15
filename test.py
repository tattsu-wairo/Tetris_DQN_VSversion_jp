import random

final_presentation_laboratory_list = ["藤田研"]
predict_laboratory_list = ["廣津研", "雪田研", "相島研", "馬　研", "李　研", "善甫研", "赤石研"]
predict_laboratory = random.choice(predict_laboratory_list)
final_presentation_laboratory_list.append(predict_laboratory)
# print(final_presentation_laboratory_list)


predict_laboratory_dict = {
    "廣津研":0,
    "雪田研":0,
    "相島研":0,
    "馬　研":0,
    "李　研":0,
    "善甫研":0,
    "赤石研":0,
}

roop = 5000000
for i in range(roop):
    predict_laboratory_dict[random.choice(predict_laboratory_list)]+=1

for i in predict_laboratory_list:
    print("{:<4}:{}".format(i,predict_laboratory_dict[i]),end="   ")
    print(predict_laboratory_dict[i]/roop*100,"%")
