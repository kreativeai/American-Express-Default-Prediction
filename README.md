# American Express Default Prediction (Kaggle Competition) - Ongoing

## Loading the data

df_train = pd.read_csv('../input/train_data.csv')

df_labels = pd.read_csv('../input/train_labels.csv')

df_train.shape

(5531451, 190)

## EDA

Check null values

![newplot](https://user-images.githubusercontent.com/78861401/171656165-f6eaa4cc-b673-4007-a892-51b4d21de9f0.png)



Distribution of Default

<img width="508" alt="Screen Shot 2022-06-02 at 10 48 16 PM" src="https://user-images.githubusercontent.com/78861401/171656867-0ee43686-2204-4b06-a1c9-cc20265709ad.png">



Statements per customer

<img width="492" alt="Screen Shot 2022-06-02 at 10 50 08 PM" src="https://user-images.githubusercontent.com/78861401/171657204-d79bab4d-2a48-4fe0-96d4-12800663ec82.png">



Categorical Features Analysis

<img width="955" alt="Screen Shot 2022-06-02 at 10 51 37 PM" src="https://user-images.githubusercontent.com/78861401/171657549-0ad19d9f-4d70-49b7-bbeb-24a0d95de41d.png">
<img width="955" alt="Screen Shot 2022-06-02 at 10 51 44 PM" src="https://user-images.githubusercontent.com/78861401/171657558-6b9b6de2-98ec-4d11-9594-2f8c1d1d5765.png">



Numerical Features Analysis

<img width="955" alt="Screen Shot 2022-06-02 at 10 53 57 PM" src="https://user-images.githubusercontent.com/78861401/171658990-9544f584-6d60-4892-874e-2343c7f1c3d5.png">
<img width="949" alt="Screen Shot 2022-06-02 at 10 54 16 PM" src="https://user-images.githubusercontent.com/78861401/171659006-97a181cb-8620-4163-8242-38f6907974c5.png">
<img width="947" alt="Screen Shot 2022-06-02 at 10 54 27 PM" src="https://user-images.githubusercontent.com/78861401/171659022-9bc1f6a8-a99a-45bb-9c0d-2e90053b8e3c.png">
<img width="947" alt="Screen Shot 2022-06-02 at 10 54 36 PM" src="https://user-images.githubusercontent.com/78861401/171659033-b45816ba-a78d-4270-9978-b116e60b9cc6.png">
<img width="948" alt="Screen Shot 2022-06-02 at 10 54 46 PM" src="https://user-images.githubusercontent.com/78861401/171659060-59e1e310-c741-4d09-8b5e-1d4b41f95069.png">
<img width="946" alt="Screen Shot 2022-06-02 at 10 54 56 PM" src="https://user-images.githubusercontent.com/78861401/171659067-710bc66d-de6c-4014-bc83-380603488a80.png">
<img width="947" alt="Screen Shot 2022-06-02 at 10 55 06 PM" src="https://user-images.githubusercontent.com/78861401/171659085-979f075b-2838-41c8-937a-8fb5b9049b61.png">
<img width="952" alt="Screen Shot 2022-06-02 at 10 58 14 PM" src="https://user-images.githubusercontent.com/78861401/171659098-6ec92778-4cf0-497b-9f9c-c4096deb80a6.png">
<img width="948" alt="Screen Shot 2022-06-02 at 10 55 26 PM" src="https://user-images.githubusercontent.com/78861401/171659105-5681a99e-4122-4468-9094-b247ff079920.png">
<img width="944" alt="Screen Shot 2022-06-02 at 10 55 16 PM" src="https://user-images.githubusercontent.com/78861401/171659108-ee041fe5-b72f-4411-b872-e51071c5ec8f.png">
<img width="945" alt="Screen Shot 2022-06-02 at 11 00 43 PM" src="https://user-images.githubusercontent.com/78861401/171659633-2dc3759d-17cc-42e6-8c5c-abe37ac96833.png">
<img width="948" alt="Screen Shot 2022-06-02 at 11 00 34 PM" src="https://user-images.githubusercontent.com/78861401/171659639-32f2bf72-b7f1-4e28-9f39-64fb016d1c74.png">
<img width="948" alt="Screen Shot 2022-06-02 at 11 00 26 PM" src="https://user-images.githubusercontent.com/78861401/171659642-491f9258-3780-4c5d-b8dd-25bdee00fb82.png">
<img width="950" alt="Screen Shot 2022-06-02 at 11 00 17 PM" src="https://user-images.githubusercontent.com/78861401/171659646-1012ea55-ab45-477a-8c36-c00ca9de5b32.png">
<img width="949" alt="Screen Shot 2022-06-02 at 11 00 08 PM" src="https://user-images.githubusercontent.com/78861401/171659650-45a51481-778a-4be4-8485-e4ff67cc8917.png">
<img width="947" alt="Screen Shot 2022-06-02 at 11 00 01 PM" src="https://user-images.githubusercontent.com/78861401/171659653-89781aa3-6dbf-4d22-b353-86c5e9f09592.png">
<img width="949" alt="Screen Shot 2022-06-02 at 10 59 51 PM" src="https://user-images.githubusercontent.com/78861401/171659655-e69ea96d-b6e3-4f0a-bb76-4e72b8c55ac0.png">
<img width="950" alt="Screen Shot 2022-06-02 at 11 01 01 PM" src="https://user-images.githubusercontent.com/78861401/171659659-01bb20bd-92cd-46ee-837e-74b59bb81953.png">
<img width="945" alt="Screen Shot 2022-06-02 at 11 01 10 PM" src="https://user-images.githubusercontent.com/78861401/171659663-20e6a32c-3ab6-4843-951b-721cb5ee4be3.png">
<img width="948" alt="Screen Shot 2022-06-02 at 11 01 19 PM" src="https://user-images.githubusercontent.com/78861401/171659665-2a91c387-1e44-42b3-955e-563bffeb5c63.png">
<img width="950" alt="Screen Shot 2022-06-02 at 11 02 17 PM" src="https://user-images.githubusercontent.com/78861401/171659919-890585c1-bf9f-413c-96b6-feeff3e6ae60.png">
<img width="948" alt="Screen Shot 2022-06-02 at 11 02 09 PM" src="https://user-images.githubusercontent.com/78861401/171659927-1df53749-c010-4f33-90e5-c2827163dbdf.png">
<img width="372" alt="Screen Shot 2022-06-02 at 11 03 30 PM" src="https://user-images.githubusercontent.com/78861401/171660342-95185e3e-8078-421f-8f8a-b6b313ede146.png">







