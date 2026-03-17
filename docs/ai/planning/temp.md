Я выполнял тестовое задание по написанию Андройд приложения по тз. Я столкнулся со следубщей проблемой, которую так и адресовал жури: --- Экран загрузки книги (файла) требует использование Firebase Storage. Сейчас даже для бесплатного использования надо привязать банковскую карту. В России это невозможно --- Я пологаю что это критический недочет жури, который они не успели предвидеть, я думаю что в таком кейсе почти никто не будет реализовывать задуманное. Однако моя цель максимально эффективно продемонстрировать свои навыки и выделиться среди всех участников. Как мне поступить в этом случае, если по факту я не могу реализовать то что необходимо, но тем не менее, я должен что то выполнить и сделать это хорошо, чтобы пройти отбор? Вот ТЗ:
---

---

Коротко: у тебя тут не техническая, а инфраструктурная блокировка. Логика задания нормальная, но в РФ реально не размашешься с биллингом Google. В такой ситуации задача — не “ныть, что нельзя сделать”, а показать инженерное мышление: обойти ограничение так, чтобы было видно, что прод сделать ты умеешь.

Разложу по опциям.

---

## 1. Идеальный вариант: Firebase Emulator Suite

Firebase официально даёт локальные эмуляторы для Auth / Firestore / Storage. Это полноценный Firebase по API, но локально, без карты и аккаунта с биллингом. ([Firebase][1])

Схема:

1. Ставишь Firebase CLI + Emulator Suite:

   * `npm install -g firebase-tools`
   * `firebase init` (отмечаешь Firestore, Storage, Auth + Emulators)
   * `firebase emulators:start`

2. В Android-проекте:

   * Подключаешь обычные Firebase SDK.
   * В debug-конфигурации указываешь использовать эмуляторы:

   ```kotlin
   // Storage
   val storage = Firebase.storage
   storage.useEmulator("10.0.2.2", 9199)

   // Firestore
   val firestore = Firebase.firestore
   firestore.useEmulator("10.0.2.2", 8080)

   // Auth
   val auth = Firebase.auth
   auth.useEmulator("10.0.2.2", 9099)
   ```

   `10.0.2.2` — стандартный адрес “host machine” из Android-эмулятора. ([Firebase][2])

3. Вся остальная логика 1:1 как с боевым Firebase:

   * Загрузка файлов в Storage (progress listener).
   * Метаданные в Firestore.
   * Auth через FirebaseAuth.

Для жюри это выглядит как:

* Код написан под **реальный Firebase**, а не под самопальный мок.
* Его легко переключить на прод: достаточно убрать `useEmulator()` и подставить нормальный `google-services.json`.

В README это формулируется примерно так:

> В связи с ограничениями оплаты Google Cloud / Firebase из РФ для реализации задания используется Firebase Local Emulator Suite (Auth, Firestore, Storage). Код клиента полностью совместим с production Firebase и требует только замены `useEmulator()` и подключения реального проекта.

Это максимально сильная позиция: ты честно говоришь про ограничение, но при этом показал, что умеешь работать с “правильным” стеком.

---

## 2. Если даже эмулятор не взлетит: абстракции + заменяемый backend

Если по какой-то причине Firebase CLI / login у тебя вообще не работает, тогда следующий по силе вариант — **чистая архитектура с заменяемыми data source’ами**.

### 2.1. Архитектура

Разделяешь слои:

* `domain`: сущности `Book`, `User`, use cases типа `UploadBook`, `GetBooks`, `DownloadBook`, `DeleteBook`.
* `data`:

  * `BookRemoteDataSource` (контракт под Firestore, Storage).
  * `BookLocalDataSource` (контракт под Room / File API).
  * `BookRepository` — агрегация логики sync локальное ↔ удалённое.
* `ui`: экраны (Compose) + ViewModel’и с состояниями.

Интерфейсы, например:

```kotlin
interface BookRemoteDataSource {
    suspend fun uploadBook(
        localUri: Uri,
        title: String,
        author: String
    ): Flow<UploadProgress> // { percent, state }
    
    suspend fun fetchBooks(userId: String): List<RemoteBookDto>

    suspend fun downloadBook(bookId: String): Flow<DownloadProgress>
}

interface BookLocalDataSource {
    suspend fun saveBookFile(bookId: String, input: InputStream): File
    suspend fun deleteBookFile(bookId: String)
    fun observeLocalBooks(): Flow<List<LocalBookEntity>>
}
```

Дальше:

* Реализуешь **`FakeBookRemoteDataSource`**, который:

  * “загружает” файл в локальную папку (например, `filesDir/books/remote/`),
  * генерит прогресс 0→100% через `Flow`,
  * выдаёт “url” вида `file://…` или `content://…`.
* Параллельно пишешь **`FirebaseBookRemoteDataSource`**, но не подключаешь его в `debug`:

  * Внутри — нормальный код по документации Firebase Storage / Firestore (uploadTask, `putFile(uri)`, `.addOnProgressListener`, `downloadUrl` и т.д.).
  * Этот модуль может быть не проверен “в бою”, но по API видно, что ты понимаешь, как это делается.

Через DI:

```kotlin
@Module
@InstallIn(SingletonComponent::class)
object DataModule {

    @Provides
    fun provideBookRemoteDataSource(): BookRemoteDataSource =
        FakeBookRemoteDataSource()
        // Для прод: FirebaseBookRemoteDataSource(firebaseStorage, firestore)
}
```

В README объясняешь:

> В условиях невозможности развернуть реальный Firebase Storage в РФ, для демонстрации реализован `FakeBookRemoteDataSource`, повторяющий поведение Firebase Storage по API (прогресс, ошибки, delayed upload). В модуле `firebase-impl` приведена реализация под Firebase SDK, которую можно подключить в production-сборке одним изменением DI.

То есть ты показываешь:

* грамотную модульность,
* готовность к прод-окружению,
* “production mindset”.

---

## 3. Как выделиться за счёт самого решения

Вне вопроса Firebase, чтобы пройти отбор, важно показать “инженера, а не кодера интерфейсов”. Я бы сделал акцент на:

### 3.1. Архитектура и состояние

* MVVM + use cases, единый `UiState` per screen, sealed классы:

```kotlin
sealed interface UploadUiState {
    object Idle : UploadUiState
    object PickingFile : UploadUiState
    data class Uploading(val percent: Int) : UploadUiState
    object Success : UploadUiState
    data class Error(val message: String) : UploadUiState
}
```

* Потоки (`Flow`/`StateFlow`) вместо коллбеков, чёткая обработка ошибок и offline.

### 3.2. Offline-first для книг

* Room (или хотя бы локальный DAO поверх файловой системы).
* При старте экрана:

  * загружаешь список из локальной БД мгновенно,
  * параллельно подтягиваешь Firestore (или fake-remote) и мерджишь.
* При удалении — сначала локальное удаление, затем синк в Firestore (если доступен).

### 3.3. UX

* Jetpack Compose + Material 3, поддержка dark/light темы.
* Нормальные состояния:

  * пустой список, “ничего не найдено”, ошибка, лоадер.
* Экран чтения:

  * шрифт [min, mid, max],
  * сохранение позиции чтения (DataStore/SharedPreferences),
  * базовые темы: светлая / тёмная / сепия.

### 3.4. Тесты

Даже пара-тройка unit-тестов на:

* `BookRepository` (merge локальное/удалённое).
* `UploadBookViewModel` (правильный переход состояний при успешной и упавшей загрузке, с использованием fake data source).

Для тестового — жирный плюс.

---

## 4. Как это корректно объяснить жюри

Я бы оформил это в README и/или сопроводительном письме максимально спокойно и по делу.

Пример формулировки (можно почти дословно):

> ### Ограничения окружения
>
> Для реализации задания требуется Firebase Storage. На момент выполнения тестового задания доступ к развертыванию Storage-проекта с бесплатным тарифом из РФ ограничен требованием привязки банковской карты зарубежного банка. Такой карты у меня нет, поэтому развернуть боевой Storage-бакет я не смог.
>
> В связи с этим:
>
> 1. Код написан под стандартные Firebase SDK (Auth, Firestore, Storage).
> 2. Для запуска и проверки используется Firebase Emulator Suite / либо fake-реализация `BookRemoteDataSource`, полностью повторяющая API Firebase Storage с прогрессом и ошибками.
> 3. Переключение на production Firebase требует только:
>
>    * указать реальные `google-services.json`,
>    * убрать `useEmulator()` / заменить `FakeBookRemoteDataSource` на `FirebaseBookRemoteDataSource` в DI.
>
> Считаю, что такой подход корректно демонстрирует мои навыки интеграции с Firebase и проектирования архитектуры, несмотря на внешние ограничения инфраструктуры.

---

## 5. Что делать прямо сейчас

Если кратко в виде плана действий:

1. Попробовать вариант с **Firebase Emulator Suite**. Если поднимется — это лучший ответ на их ТЗ.
2. Параллельно заложить **интерфейсы и DI**, чтобы backend можно было менять (fake ↔ Firebase).
3. Реализовать весь UI и поведение экранов строго по ТЗ:

   * состояния, ошибки, retry, прогресс, поиск, чтение книги.
4. Написать README с:

   * архитектурой,
   * описанием ограничений,
   * инструкцией, как переключить на real Firebase.
5. Приложить скриншоты / короткое видео работы приложения.

Если хочешь, дальше можем разобрать конкретно архитектуру модулей и контракты репозиториев под это ТЗ, чтобы их сразу спроектировать “под прод + эмулятор/фейк”.

[1]: https://firebase.google.com/docs/emulator-suite?utm_source=chatgpt.com "Introduction to Firebase Local Emulator Suite"
[2]: https://firebase.google.com/docs/emulator-suite/connect_storage?utm_source=chatgpt.com "Connect your app to the Cloud Storage for Firebase Emulator"

---

Если я правильно понял:

я ставлю эмулятор firebase, пишу код абсолютно так же как если бы был подключен фаербейз, но вместо реального фаербейз юзаю его и никаких проблем?

---


В общих чертах ты понял верно, но есть нюансы.

По сути схема такая:

1. Пишешь код так, **как будто работаешь с обычным Firebase**:

   * `FirebaseAuth` для логина,
   * `FirebaseStorage` для загрузки файлов,
   * `FirebaseFirestore` для метаданных.
2. Но в `debug`-сборке говоришь SDK: “ходи не в облако, а в локальный эмулятор”.

То есть API и архитектура такие же, как для реального проекта. Отличается только один небольшой кусок конфигурации.

---

## Что именно нужно сделать

### 1. Поднять эмуляторы

На машине разработки:

```bash
npm install -g firebase-tools
firebase login           # если получится
firebase init emulators  # выбираешь: Auth, Firestore, Storage
firebase emulators:start
```

Получишь локальные порты, по умолчанию примерно:

* Auth: `9099`
* Firestore: `8080`
* Storage: `9199`

(порты можно настроить в `firebase.json`).

Да, всё это крутится локально, никакой карты и биллинга тут не задействовано.

### 2. Настроить Android-клиент

В Android-приложении всё как обычно:

* Подключаешь зависимости Firebase (BOM + `firebase-auth`, `firebase-firestore`, `firebase-storage`).
* Добавляешь `google-services.json` (его удобнее всё равно получить из реального Firebase-проекта, но это не связано с биллингом/Storage, проект можно создать и без карты).

Дальше в `Application` или DI-модуле для `debug`:

```kotlin
class App : Application() {
    override fun onCreate() {
        super.onCreate()

        if (BuildConfig.DEBUG) {
            Firebase.firestore.useEmulator("10.0.2.2", 8080)
            Firebase.storage.useEmulator("10.0.2.2", 9199)
            Firebase.auth.useEmulator("10.0.2.2", 9099)
        }
    }
}
```

Важно:

* `10.0.2.2` — это `localhost` твоей машины из Android-эмулятора.
* В `release` это условие не сработает, и SDK будет ходить в нормальный облачный Firebase (если его подключат).

### 3. Весь остальной код — “как по ТЗ”

Экран загрузки:

```kotlin
val storageRef = Firebase.storage.reference
val fileRef = storageRef.child("books/$uid/${fileName}")

val uploadTask = fileRef.putFile(fileUri)

uploadTask.addOnProgressListener { taskSnapshot ->
    val progress = (100.0 * taskSnapshot.bytesTransferred / taskSnapshot.totalByteCount).toInt()
    // обновляешь UiState.Uploading(progress)
}.addOnSuccessListener {
    fileRef.downloadUrl.addOnSuccessListener { uri ->
        firestore.collection("books").add(
            mapOf(
                "title" to title,
                "author" to author,
                "fileUrl" to uri.toString(),
                "userId" to uid
            )
        )
    }
}.addOnFailureListener {
    // UiState.Error(...)
}
```

Это ровно тот же код, который будет работать и с реальным Firebase, потому что эмулятор полностью совместим по API.

---

## Как это подать жюри

Ключевой момент — явно проговорить в README:

* Используется **Firebase Emulator Suite** (Auth, Firestore, Storage) по причинам недоступности биллинга из РФ.
* Клиентский код написан под **production Firebase**:

  * переключение между эмулятором и боем — это только `useEmulator(...)` + конфиг.
* Архитектурно всё уже подготовлено под реальный клауд: есть нормальный `Repository`, обработка прогресса, ошибок, offline и т.д.

Тогда ответ на твой вопрос:

> я ставлю эмулятор firebase, пишу код абсолютно так же как если бы был подключен фаербейз, но вместо реального фаербейз юзаю его и никаких проблем?

Да, концептуально так.
Практически: “никаких проблем” при условии, что:

* эмуляторы реально подняты и доступны по сети (порты, `10.0.2.2`);
* ты не тащишь `useEmulator()` в релизную конфигурацию;
* в README честно описано окружение и как перевести это в боевой Firebase.

Если хочешь, дальше можем наметить структуру модулей/пакетов под это ТЗ с учётом эмуляторов и DI.
