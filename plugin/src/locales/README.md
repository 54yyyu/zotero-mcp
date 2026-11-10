# The panel's languages

One catalog per language: `en.ts` is the source and the fallback, the others are typed `Record<Key, Msg>` against it.
How it works is in DESIGN.md "Languages"; `test/i18n.test.ts` checks keys, placeholders, `code` spans and plural forms.

## Rules for every catalog

- Zotero's own translation for every Zotero concept (from Zotero's locale files: `unzip` `chrome/locale/<loc>/zotero/*.ftl`
  and `zotero.properties` out of `Zotero.app/.../omni.ja`). The tables below are what each catalog uses.
- `{name}` placeholders and text in backticks stay exactly as in English; product, model, command and file names are not
  translated.
- Short: a button or label at most about 30% longer than the English. Zotero's tone in that language (formal address
  where Zotero uses it). The default prompts (`prompt.*.text`) are the user's own voice, as a researcher would type.
- A plural message has the language's own categories only (`Intl.PluralRules`): `other` for Chinese and Japanese, `one`
  and `other` for the rest.

## Adding a language

Add `<id>.ts` (copy `en.ts`, translate every value), its id to `UI_LANGUAGES` and `STARTUP_WORDS` in `languages.ts`, the
locale mapping in `uiLanguageFor` if Zotero's locale is not `<id>-*`, the import in `src/i18n.ts`, and a glossary here.
Mark it `beta: true` until a native speaker has read it in the panel.

## Adding a string

Add the key to `en.ts` in its group, then to every other catalog (the type check names the ones missing). Use `t()` where
the element is drawn, never in a module constant: the language can change while the panel is open.

## Simplified Chinese (zh-Hans): for the maintainer's review

Address the user as 您, never 你, as Zotero's own Chinese UI does.

| English (Zotero term) | Simplified Chinese (zh-Hans) |
|---|---|
| Item | 条目 |
| Collection | 分类 |
| Library | 文献库 |
| Annotation | 注释 |
| Note / Attachment / Tag | 笔记 / 附件 / 标签 |
| Reader | 阅读器 |
| Highlight / Underline / Image / Ink | 高亮 / 下划线 / 图片 / 笔迹 |
| Settings > Advanced | 设置 > 高级 |
| Select Area (chips: Selected Area / Text Selection) | 选择区域（选中区域 / 选中文本） |
| Show in Library / Copy / Save / Delete / Remove | 在文献库中显示 / 复制 / 保存 / 删除 / 移除 |
| Appearance / Density / Compact / Comfortable / Automatic / Page | 外观 / 密度 / 紧凑 / 舒适 / 自动 / 页 |
| Agent | 智能体 |
| Chat (New chat) | 对话（新对话） |
| Prompt | 提示词 |
| Skill | 技能 |
| Model / Effort | 模型 / 推理强度 |
| Permission mode | 权限模式 |
| Context window | 上下文窗口 |
| Token | token |
| Subscription / Sign in | 订阅 / 登录 |
| API key | API 密钥 |
| Summarise now (compact) | 立即压缩 |
| Recheck | 重新检查 |
| Pin / Unpin | 固定 / 取消固定 |
| Folder / File / Terminal / Keychain | 文件夹 / 文件 / 终端 / 钥匙串 |

## Traditional Chinese (zh-Hant, Taiwan usage): for the maintainer's review

Address the user as 您, never 你, as Zotero's own Chinese UI does.

| English (Zotero term) | Traditional Chinese (zh-Hant) |
|---|---|
| Item | 項目 |
| Collection | 文獻集 |
| Library | 文獻庫 |
| Annotation | 標註 |
| Note / Attachment / Tag | 筆記 / 附件 / 標籤 |
| Reader | 閱讀器 |
| Highlight / Underline / Image / Ink | 醒目提示 / 底線 / 圖片 / 手繪 |
| Settings > Advanced | 設定 > 進階 |
| Select Area (chips: Selected Area / Text Selection) | 選擇區域（選取區域 / 選取文字） |
| Show in Library / Copy / Save / Delete / Remove | 在文獻庫中顯示 / 複製 / 儲存 / 刪除 / 移除 |
| Appearance / Density / Compact / Comfortable / Automatic / Page | 外觀 / 密度 / 緊緻 / 舒適 / 自動 / 頁 |
| Agent | AI 代理 |
| Chat (New chat) | 對話（新對話） |
| Prompt | 提示詞 |
| Skill | 技能 |
| Model / Effort | 模型 / 推理強度 |
| Permission mode | 權限模式 |
| Context window | 上下文視窗 |
| Token | token |
| Subscription / Sign in | 訂閱 / 登入 |
| API key | API 金鑰 |
| Summarise now (compact) | 立即壓縮 |
| Recheck | 重新檢查 |
| Pin / Unpin | 釘選 / 取消釘選 |
| Folder / File / Terminal / Keychain | 資料夾 / 檔案 / 終端機 / 鑰匙圈 |

## Japanese (ja), beta: awaiting a native speaker's review

| English (Zotero term) | 日本語 |
|---|---|
| Item (Zotero) | 項目 |
| Collection (Zotero) | コレクション |
| Annotation (Zotero) | 注釈 |
| Note (Zotero) | メモ |
| Library / My Library (Zotero) | ライブラリ / マイライブラリ |
| Attachment (Zotero) | 添付ファイル |
| Reader (Zotero) | リーダー |
| Highlight / Underline / Image / Ink (Zotero) | 強調表示 / 下線 / 画像 / インク |
| Settings > Advanced (Zotero) | 「設定」>「詳細」 |
| Allow other applications on this computer to communicate with Zotero (Zotero) | このコンピュータ上の他のアプリケーションが Zotero と通信できるようにする |
| Density / Compact / Comfortable (Zotero) | 項目間の間隔 / コンパクト / 標準 |
| Delete / Remove / Cancel / Save / Done (Zotero) | 削除 / 削除 / キャンセル / 保存 / 完了 |
| Agent | エージェント |
| Chat / New chat | チャット / 新しいチャット |
| Prompt / Skill | プロンプト / スキル |
| Effort | 推論レベル |
| Permission mode | 権限モード |
| Context window | コンテキストウィンドウ |
| Model / Token | モデル / トークン |
| Subscription | サブスクリプション |
| API key | API キー |
| Sign in | サインイン |
| Summarise now (compact) | 今すぐ要約 |
| Recheck | 再確認 |
| Pin / Unpin | ピン留め / ピン留めを解除 |

## German (de), beta: awaiting a native speaker's review

| English (Zotero term) | Deutsch |
|---|---|
| Item | Eintrag |
| Collection | Sammlung |
| Annotation | Anmerkung |
| Note | Notiz |
| Library | Bibliothek |
| Attachment | Anhang |
| Tag | Tag |
| Reader | Reader |
| Settings > Advanced | Einstellungen > Erweitert |
| Highlight / Underline (annotation types) | Markieren / Unterstrichen (as in Zotero's reader) |
| Note annotation | Fußnote (as in Zotero's reader) |
| Density: Compact / Comfortable | Dichte: Kompakt / Locker |
| Appearance | Erscheinungsbild |
| Agent | Agent (Dativ/Akkusativ: Agenten) |
| Chat | Chat |
| Prompt | Prompt |
| Skill | Skill |
| Model | Modell |
| Effort | Aufwand |
| Permission mode | Berechtigungsmodus |
| Context window | Kontextfenster |
| Token | Token |
| Subscription | Abo |
| API key | API-Schlüssel |
| Sign in | Anmelden / Anmeldung |
| Summarise now | Jetzt zusammenfassen |
| Recheck | Erneut prüfen |
| Pin / Unpin | Anheften / Anheftung aufheben |
| Panel | Seitenleiste |
| Addressing the user | Sie (as in Zotero's German UI) |

## French (fr), beta: awaiting a native speaker's review

| English (Zotero term) | Français |
|---|---|
| Item | document |
| Collection | collection |
| Annotation | annotation |
| Note | note |
| Library | bibliothèque |
| Attachment | pièce jointe |
| Tag | marqueur |
| Highlight (annotation type) | Surlignage (Zotero's tool: Surligner) |
| Underline | Souligné |
| Ink | Annotation manuscrite |
| Settings > Advanced | Paramètres > Avancé |
| Save / Cancel / Delete / Remove | Enregistrer / Annuler / Supprimer / Supprimer (context chips: Retirer) |
| Text Selection / Selected Area | Sélection de texte / Zone sélectionnée |
| agent | agent |
| chat (a conversation) | conversation |
| prompt | prompt |
| skill | compétence |
| model / effort | modèle / effort |
| permission mode | mode d'autorisation |
| context window | fenêtre de contexte |
| token | jeton |
| subscription / API key | abonnement / clé API |
| sign in | se connecter (card: Connexion) |
| Summarise now (compact) | Résumer maintenant |
| Recheck | Revérifier |
| Pin / Unpin | Épingler / Désépingler |

## Spanish (es), beta: awaiting a native speaker's review

| English (Zotero term) | Español |
|---|---|
| item | elemento |
| collection | colección |
| annotation | anotación |
| note | nota |
| library | biblioteca |
| attachment | adjunto |
| tag | etiqueta |
| Reader | lector |
| Highlight / Underline / Image / Ink | Destacado / Subrayado / Imagen / Tinta |
| Settings | Ajustes |
| Advanced | Avanzadas |
| Delete / Remove | Borrar / Eliminar |
| Done / Cancel / Save / Copy / Edit | Hecho / Cancelar / Guardar / Copiar / Editar |
| Density / Compact / Comfortable | Densidad / Compacto / Confortable |
| Appearance | Apariencia |
| Allow / Always allow / Deny | Permitir / Permitir siempre / Negar |
| agent | agente |
| chat | chat |
| prompt | prompt |
| skill | habilidad |
| model | modelo |
| effort | esfuerzo |
| permission mode | modo de permisos |
| context window | ventana de contexto |
| token | token |
| subscription | suscripción |
| API key | clave de API |
| sign in | iniciar sesión |
| Summarise now | Resumir ahora |
| Recheck | Comprobar de nuevo |
| pin / unpin | fijar / dejar de fijar |
| Text Selection / Selected Area | Texto seleccionado / Área seleccionada |

## Brazilian Portuguese (pt-BR), beta: awaiting a native speaker's review

| English (Zotero term) | Português (Brasil) |
|---|---|
| item | item |
| collection | coleção |
| annotation | anotação |
| note | nota |
| library | biblioteca |
| attachment | anexo |
| tag | etiqueta |
| Reader | leitor |
| Highlight / Underline / Image / Ink | Destaque / Sublinhado / Imagem / Tinta |
| Settings | Configurações |
| Advanced | Avançado |
| Delete / Remove | Remover / Remover |
| Done / Cancel / Save / Copy / Edit | Feito / Cancelar / Salvar / Copiar / Editar |
| Density / Compact / Comfortable | Densidade / Compacto / Confortável |
| Appearance | Aparência |
| Allow / Always allow / Deny | Permitir / Sempre permitir / Negar |
| agent | agente |
| chat | chat |
| prompt | prompt |
| skill | habilidade |
| model | modelo |
| effort | esforço |
| permission mode | modo de permissão |
| context window | janela de contexto |
| token | token |
| subscription | assinatura |
| API key | chave de API |
| sign in | entrar |
| Summarise now | Resumir agora |
| Recheck | Verificar de novo |
| pin / unpin | fixar / desafixar |
| Text Selection / Selected Area | Texto selecionado / Área selecionada |
