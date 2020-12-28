# mmp-practice-hm3
Практическое задание студентов каферды ММП на факультете ВМК МГУ.
1. В папке figures лежат исходники изображений для отчета
2. В папке data лежат исходники данных для эспериментов и для тестирования сервера
3. В папке report tex и pdf версии отчета
4. В файле experiments.ipynb проведены все необходимые эксперименты
5. В файле ensemples.py реализованы необходимые алгоритмы
## Flask Server 
1. Папка data должна монтироваться в докер контейнер для хранения результатов, датасетов и модели(естественно, необязательно именно эту папку монтировать, но
важно монтировать в путь "/root/FlaskServer/data".
2. src содержит всю необходимую логику работы серевера.
    + main_server.py ядро сервера
    + models.py дублирование реализаций алгоритмов из практического задания  
## Docker
Docker образ https://hub.docker.com/repository/docker/vladtytskiy/mmp-practice-hm3
1. Чтобы собрать докер образ: `docker build -t repo_name/image_name:image_tag .`
2. Чтобы его запустить: `docker run -p 5000:5000 -v "$PWD/FlaskServer/data:/root/FlaskServer/data" --rm -i repo_name/image_name`
## Инструкция к применению
Это максимально простой и урезанный сервер, который только можно представить. Все действия линейны и интуитивны.  
В папке `/root/FlaskServer/data` будут сохраняться модели, датасеты и предсказанные значения на тестe.  
Каждый датасет(и тестовый) должен быть в формате csv и содержать колонку, которая является тагретом.

## Вздохи грусти и печали
К сожалению из-за других дел перед Новым годом я не успел сделать НИЧЕГО, что хотел бы сам видеть в таком задании. Пришлось делать на коленке за один день. Но это было очень интересно и познавательно. Спасибо за отличное задание!

## P.S. 
**ЕСЛИ ЧТО-ТО НЕ РАБОТАЕТ, НЕ БЕЙТЕ И СВЯЖИТЕСЬ СО МНОЙ `telegram: @v_tytskiy`.**  
Любой порядочный тестировщик, да и просто адекватный человек, с радостью попрыгал бы на моих костях, если бы увидел этот код.
