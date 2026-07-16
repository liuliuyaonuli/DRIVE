#!/usr/bin/env python3
"""
Site profiles for trajectory-to-skill generation.

Each profile describes the site-specific vocabulary, selectors, URL patterns,
and task families needed by the shared skill generator and pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).parent


@dataclass(frozen=True)
class SiteSkillProfile:
    site: str
    display_name: str
    platform_description: str
    analysis_script: Path
    task_families: List[str]
    verified_selectors: str
    url_patterns: str
    success_generation_guidance: str
    failure_generation_guidance: str
    example_skill_patterns: str
    operation_failure_examples: List[str]
    reasoning_failure_examples: List[str]
    family_keywords: Dict[str, List[str]]
    object_keywords: Dict[str, List[str]]
    default_object: str
    expected_states: Dict[str, str]
    verification_strategies: Dict[str, List[str]]


def _script(name: str) -> Path:
    return SCRIPT_DIR / name


SITE_PROFILES: Dict[str, SiteSkillProfile] = {
    "gitlab": SiteSkillProfile(
        site="gitlab",
        display_name="GitLab",
        platform_description="GitLab CE project management and source-code collaboration site.",
        analysis_script=_script("analyze_gitlab_failures_v2.py"),
        task_families=[
            "Repo_Lifecycle",
            "Issue_Management",
            "Commit_Analysis",
            "Merge_Request",
            "Collaboration_Access",
            "Content_Editing",
            "Profile_Settings",
        ],
        verified_selectors="""
Navigation and search:
- Global search: `input[placeholder*='Search'], input[aria-label*='Search'], input[type='search']`
- Project links: `a[href*='/-/'], a[href^='/']`
- New buttons: `a:has-text('New'), button:has-text('New')`

Repository/project:
- Project sidebar links: `a:has-text('Issues')`, `a:has-text('Merge requests')`, `a:has-text('Repository')`
- Star button: `button:has-text('Star'), a:has-text('Star')`
- Fork button: `a:has-text('Fork'), button:has-text('Fork')`
- Files table rows: `.tree-item, tr:has(a)`

Issues and merge requests:
- Issue/MR rows: `li.issue, li.merge-request, .issuable-list li`
- Title input: `input[name='issue[title]'], input[name='merge_request[title]'], input[name='title']`
- Description textarea: `textarea[name*='description'], textarea`
- Labels/assignees: `button:has-text('Label'), button:has-text('Assignee'), input[placeholder*='Search']`
- Submit buttons: `button:has-text('Create issue')`, `button:has-text('Create merge request')`, `button:has-text('Save')`

Filters and pagination:
- Filter input: `input[placeholder*='Filter'], input[aria-label*='Filter']`
- Author/date/filter chips: `.gl-filtered-search-token, .filtered-search-token`
- Next page/load more: `a[rel='next'], button:has-text('Load more')`
""",
        url_patterns="""
- Project root: `/{namespace}/{project}`
- Issues list: `/{namespace}/{project}/-/issues`
- Issue detail: `/{namespace}/{project}/-/issues/{issue_id}`
- Merge requests: `/{namespace}/{project}/-/merge_requests`
- Commits: `/{namespace}/{project}/-/commits/{branch}`
- Repository files: `/{namespace}/{project}/-/tree/{branch}`
- User profile: `/{username}`
""",
        success_generation_guidance="""
GitLab-specific skill patterns:
- Generate skills around durable GitLab objects: projects, issues, merge requests,
  commits, files, members, and profile settings.
- Prefer parameters like `namespace`, `project_name`, `issue_title`,
  `branch_name`, `username`, `label_name`, and `target_role`.
- Use the following GitLab task patterns as guidance, not as rigid routing rules:
  * Repository and project lifecycle: find public projects, get an SSH clone command,
    fork_project repositories, star top repositories, create empty or template-based
    public/private projects, add initial members, edit LICENSE content, and update a
    project site title or project homepage setting.
  * Issue and task management: browse recent or open issues, apply issue filters by
    label/title/status/date, find the latest updated or created issue by keyword,
    create issues with assignee and due date, open todos, and create a milestone with
    start/end dates.
  * Commit history and contribution analysis: count commits by author/date/repository,
    paginate commit lists, identify the top contributor or top contributors, extract
    contributor email addresses, and combine contribution queries with repository star
    thresholds.
  * Merge request and code review workflows: open merge requests assigned to me,
    open merge requests requiring my review, locate MRs by title or branch, post
    review comments, verify whether discussion comments were resolved, and create
    merge requests with source branch, target branch, and reviewer selection.
  * Collaboration, permissions, groups, and follows: inspect who has access to a
    project, invite users as guest/reporter/developer/maintainer, add multiple
    members, create a new group with members, and follow GitLab users.
  * Repository content and documentation editing: create folders and files in the
    repository tree, write structured link lists, create README files for new projects,
    and preserve exact requested content when editing repository files.
  * Profile and account settings: set GitLab status text, update the profile homepage
    URL, and retrieve account integration values such as the RSS feed token.
- For list/filter tasks, keep the browser on the required list page when the
  task is evaluated by URL; do not click into detail pages unless the task asks.
- When filtering issues or commits, verify issue filters or commit filters by
  checking both URL query state and visible filtered-search tokens.
- For create/edit tasks, verify the new issue/MR/file/member state after saving.
- Cover common GitLab WebArena tasks beyond CRUD: todos, SSH clone command
  lookup, starring top repositories, following accounts, milestones with dates,
  project site title/homepage settings, private template repositories, and
  inviting users with exact access roles.
""",
        failure_generation_guidance="""
GitLab-specific corrective rules:
- If the failure was an issue filters or commit filters problem, produce a skill
  that applies the filter, waits for the grid/list to update, and verifies token
  text plus result rows before returning.
- If a lifecycle task failed, verify the clone command, fork/star state, created
  project visibility/template, LICENSE content, or project site title after the
  operation.
- If an issue, todo, or milestone task failed, preserve the required issue list
  URL when appropriate and verify title, label, assignee, due date, state, or
  milestone dates after creating or filtering.
- If a commit or contributor analysis failed, scan all relevant commit pages,
  apply author/date/branch filters carefully, and return the requested count,
  top contributor, email, or star-threshold aggregation.
- If a merge request workflow failed, locate the MR by branch/title, verify
  assigned/review-required status, post the requested discussion text, or create
  the MR with the exact source branch, target branch, and reviewer.
- If a permissions/group/follow task failed, verify every username and role in
  the members or group table, or verify the follow state on each user profile.
- If a repository content task failed, reopen the file or README after saving and
  verify exact folder path, filename, and requested content.
- If the task is multi-item, such as fork_project for every repository, generate
  a complete loop over all matching project links and verify each resulting repo.
- Avoid over-navigation: preserve list URLs for tasks whose expected evaluator
  checks `/issues`, `/merge_requests`, or `/commits` list pages.
- For form submission failures, wait for the post-submit URL/detail page and
  verify the title, branch, role, or file content.
- For status/profile/project settings failures, save the setting and revisit the
  profile/project page to verify the visible value.
""",
        example_skill_patterns="""
Example GitLab skill patterns:
- `async def filter_project_issues(page, namespace: str, project_name: str, label_name: str, state: str = "opened") -> list[str]`
  Navigates to `/{namespace}/{project_name}/-/issues`, applies issue filters,
  waits for the list to update, and returns visible issue titles.
- `async def fork_project(page, namespace: str, project_name: str, target_namespace: str) -> str`
  Opens the project, starts Fork, selects namespace, submits, and verifies the
  forked project URL exists.
- `async def count_commits_by_author_on_date(page, namespace: str, project_name: str, author: str, date_text: str) -> int`
  Opens commits, applies author/date constraints, paginates, and counts rows.
- `async def invite_project_member(page, namespace: str, project_name: str, username: str, role: str) -> bool`
  Opens project members, searches the exact user, selects the requested role,
  submits, and verifies the member row.
""",
        operation_failure_examples=[
            "filter_not_applied",
            "pagination_incomplete",
            "async_not_ready",
            "selector_miss",
            "form_submit_failed",
            "over_navigation",
            "multi_step_incomplete",
            "verification_skipped",
        ],
        reasoning_failure_examples=[
            "wrong_date_format",
            "wrong_author",
            "wrong_project",
            "wrong_metric",
            "premature_stop",
            "miscount",
            "wrong_page_section",
            "ignored_constraints",
        ],
        family_keywords={
            "Repo_Lifecycle": ["repo", "repository", "project", "fork", "star", "license"],
            "Issue_Management": ["issue", "label", "milestone", "assignee", "todo"],
            "Commit_Analysis": ["commit", "author", "branch", "date", "contributor"],
            "Merge_Request": ["merge", "request", "review", "branch"],
            "Collaboration_Access": ["member", "role", "group", "permission", "access"],
            "Content_Editing": ["file", "readme", "folder", "edit", "content"],
            "Profile_Settings": ["profile", "status", "settings", "token"],
        },
        object_keywords={
            "repository": ["repository", "repo", "project"],
            "issue": ["issue", "bug", "ticket"],
            "merge_request": ["merge request", "mr"],
            "commit": ["commit", "branch", "contributor"],
            "member": ["member", "user", "role", "access"],
            "file": ["file", "readme", "folder"],
        },
        default_object="project_items",
        expected_states={
            "Repo_Lifecycle": "The repository action is visible in the project UI.",
            "Issue_Management": "The issue list or issue detail page reflects the requested change.",
            "Commit_Analysis": "The final answer reports the requested commit information in the expected format.",
            "Merge_Request": "The merge request page reflects the requested creation, update, or review action.",
            "Collaboration_Access": "The member, role, or access state is visible in GitLab.",
            "Content_Editing": "The file tree or file content shows the requested edit.",
            "Profile_Settings": "The profile or settings page displays the requested value.",
        },
        verification_strategies={
            "Repo_Lifecycle": ["Verify the project URL and action state after the operation."],
            "Issue_Management": ["Verify labels, assignees, status, and URL filters on the issue page."],
            "Commit_Analysis": ["Check all pages of commits before counting or answering."],
            "Merge_Request": ["Verify the MR title, source/target branches, and assignees/reviewers."],
            "Collaboration_Access": ["Verify the target username and role in the members table."],
            "Content_Editing": ["Open the file after saving and verify the content changed."],
            "Profile_Settings": ["Reload settings/profile and verify the saved value."],
        },
    ),
    "shopping": SiteSkillProfile(
        site="shopping",
        display_name="Shopping",
        platform_description="OneStopShop customer-facing e-commerce storefront.",
        analysis_script=_script("analyze_shopping_failures_v2.py"),
        task_families=[
            "Product_Search_Info",
            "Reviews_Ratings",
            "Cart_Wishlist_Account",
            "Order_Status",
            "Order_Statistics",
            "Aftermarket_Service",
        ],
        verified_selectors="""
Navigation and search:
- Search input: `input[name='q'], input[type='search'], input[placeholder*='Search']`
- Search submit: `button[type='submit'], button:has-text('Search')`
- Category links: `a[href*='cat='], a:has-text(category_name)`

Product lists:
- Product cards: `.product-item, li.product-item, .product, [data-product-id]`
- Product links: `a.product-item-link, a[href*='/product/'], a[href*='/catalog/product/view']`
- Price text: `.price, [data-price-type], .price-wrapper`
- Sort selector: `select[aria-label*='Sort'], select[name*='sort']`
- Filter checkboxes: `input[type='checkbox'], a[href*='price='], a[href*='brand=']`

Product detail/cart:
- Add to cart: `button:has-text('Add to Cart'), button[title='Add to Cart']`
- Add to wishlist: `a:has-text('Add to Wish List'), button:has-text('Wishlist')`
- Quantity input: `input[name='qty'], input.qty`
- Cart link: `a:has-text('Cart'), a[href*='checkout/cart']`
- Cart rows: `.cart.item, tr.item-info`

Account/orders/reviews:
- Account link: `a:has-text('My Account'), a[href*='customer/account']`
- Orders: `a:has-text('My Orders'), a[href*='sales/order/history']`
- Order rows: `tr, .order-item`
- Review textarea: `textarea, textarea[name*='review']`
- Submit buttons: `button:has-text('Submit'), button:has-text('Save'), button:has-text('Place Order')`
""",
        url_patterns="""
- Home/search: `/`, `/catalogsearch/result/?q={query}`
- Category: `/catalog/category/view/`, category URLs with query filters
- Product detail: product URLs containing `/product/` or `/catalog/product/view`
- Cart: `/checkout/cart`
- Account: `/customer/account`
- Orders: `/sales/order/history`
""",
        success_generation_guidance="""
Shopping-specific skill patterns:
- Generate skills around storefront workflows: product search, product cards,
  category filters, product variants, reviews, cart/wishlist, account fields,
  and order history.
- Prefer parameters like `product_name`, `category_name`, `brand_name`,
  `max_price`, `rating`, `order_number`, `review_keyword`, and `quantity`.
- Use the following Shopping task patterns as guidance, not as rigid routing
  rules:
  * Product search and product information: browse categories and brands, sort
    listings by price, search keywords, filter under a budget, compare models,
    compute a price range, find the most/least expensive item, satisfy constraints
    such as minimum storage capacity, and handle needs-based searches such as
    jaw bruxism or choosing storage for a number of cards.
  * Reviews and reputation analysis: open reviews, find reviewers by keyword,
    star rating, title, complaint, praise, or manufacturer/product type; extract
    customer names, review titles, relevant sentences, main criticisms, and
    summaries of what customers say; use review constraints such as at least 5 reviews
    when selecting a highest-rated or least-expensive qualifying product.
  * Cart, wish list, and account operations: add a product to the wish list,
    add selected products to cart, compare open tabs by per-unit price, update
    delivery address or account address, reorder a canceled past purchase, and
    rate a recent purchase with a nickname and star count.
  * Order status and order details: open My Orders, locate the latest order or
    most recent order by status, open a specific order number, and read status,
    delivery date, total cost, tracking, or other requested order fields.
  * Order and spending statistics: traverse order history across pages, count
    fulfilled orders, aggregate spending by time period or category, and find
    first/last purchase dates for product descriptions.
  * Aftermarket service, refund, and customer support: calculate expected refund
    amounts with or without shipping, open contact us flows, draft refund/coupon
    messages with Don't submit yet behavior, find customer-service phone numbers,
    and subscribe to the newsletter when requested.
- For product-list tasks, operate on product cards directly and avoid opening a
  detail page when the evaluator expects the filtered list page URL.
- Always verify product identity with title plus constraints such as brand,
  SKU, variant, price range, rating, or stock before interacting.
- For cart mutations, perform cart verification by opening `/checkout/cart` and
  checking the row title/quantity, not just the button click.
- Cover common Shopping WebArena tasks: reviewer extraction by phrase/rating,
  order totals over date windows, product option/config lookup from past orders,
  wishlist add, reorder cancelled purchases, buy highest-rated item within a
  budget, compare per-unit prices across open tabs, and contact/refund drafts
  that must be filled but not submitted.
""",
        failure_generation_guidance="""
Shopping-specific corrective rules:
- If a filter/search failed, generate a skill that submits the query, waits for
  product cards, verifies URL/result text, and paginates or loads more if needed.
- If a product-information workflow failed, verify product name, brand/category,
  sort order, price range, storage/spec constraints, discount, and visible card
  text before selecting or answering.
- If cart verification failed, open the cart after adding/updating/removing and
  verify the expected row, quantity, and price state.
- If the failure involved reviews, open the reviews section, apply star/content
  constraints, and verify reviewers, rating text, review titles, or extracted
  criticism/praise sentences before returning.
- If a wish list, reorder, rating, or address-update task failed, navigate to the
  relevant wishlist/account/order page and verify the changed item, field, or
  submitted review state.
- If an order-detail task failed, locate the exact order number or latest order
  with the requested status before reading total cost, shipment, arrival, or
  tracking fields.
- For order/statistics tasks, iterate order pages completely before computing
  totals or choosing earliest/latest records.
- For "do not submit" contact/refund tasks, fill the form and return a summary
  of field values without clicking the final submit button.
- For refund and service tasks, compute refund amounts from order totals and
  shipping rules, open contact us when needed, preserve Don't submit yet behavior,
  and verify customer-service phone or newsletter state when requested.
""",
        example_skill_patterns="""
Example Shopping skill patterns:
- `async def search_products_with_constraints(page, query: str, brand_name: str | None = None, max_price: float | None = None) -> list[dict]`
  Searches, applies filters, reads product cards, and returns matched products.
- `async def add_product_to_cart_and_verify(page, product_name: str, quantity: int = 1) -> bool`
  Finds the exact product, adds it to cart, opens cart, and verifies cart
  verification state.
- `async def extract_order_status(page, order_number: str) -> str`
  Opens order history, filters or scans order rows, opens the right order, and
  returns the visible status.
- `async def draft_refund_contact_message(page, product_name: str, reason: str) -> dict`
  Finds the order, computes the refundable amount, fills the contact form, and
  verifies the draft fields without submitting.
""",
        operation_failure_examples=[
            "filter_not_applied",
            "search_not_executed",
            "pagination_incomplete",
            "async_not_ready",
            "selector_miss",
            "cart_operation_failed",
            "form_submit_failed",
            "verification_skipped",
        ],
        reasoning_failure_examples=[
            "wrong_product_selected",
            "wrong_calculation",
            "wrong_order_selected",
            "premature_conclusion",
            "wrong_page_section",
            "ignored_constraints",
            "miscount",
            "wrong_answer_format",
        ],
        family_keywords={
            "Product_Search_Info": ["search", "product", "price", "brand", "category", "sort"],
            "Reviews_Ratings": ["review", "rating", "stars", "author", "feedback"],
            "Cart_Wishlist_Account": ["cart", "wishlist", "account", "address", "quantity"],
            "Order_Status": ["order", "status", "shipment", "delivery", "tracking"],
            "Order_Statistics": ["order", "spending", "count", "total", "statistics"],
            "Aftermarket_Service": ["refund", "return", "contact", "service", "newsletter"],
        },
        object_keywords={
            "product": ["product", "item", "sku"],
            "review": ["review", "rating", "stars"],
            "cart": ["cart", "basket", "quantity"],
            "order": ["order", "shipment", "delivery"],
            "account": ["account", "address", "profile"],
        },
        default_object="shopping_items",
        expected_states={
            "Product_Search_Info": "The requested product information is found and answered with all constraints satisfied.",
            "Reviews_Ratings": "The requested review or rating information is visible and extracted accurately.",
            "Cart_Wishlist_Account": "The cart, wishlist, or account page reflects the requested change.",
            "Order_Status": "The selected order details show the requested status or value.",
            "Order_Statistics": "The answer aggregates the correct orders over the requested scope.",
            "Aftermarket_Service": "The service, refund, contact, or form state matches the objective.",
        },
        verification_strategies={
            "Product_Search_Info": ["Verify product title, price, brand, and category before answering."],
            "Reviews_Ratings": ["Verify rating filters and review authors/content before stopping."],
            "Cart_Wishlist_Account": ["Open cart/wishlist/account page and verify the target item or field."],
            "Order_Status": ["Verify the order number/date/status on the order detail page."],
            "Order_Statistics": ["Iterate all relevant order pages before computing totals."],
            "Aftermarket_Service": ["Verify form fields are filled and do not submit if the task says not to submit."],
        },
    ),
    "shopping_admin": SiteSkillProfile(
        site="shopping_admin",
        display_name="Shopping Admin",
        platform_description="Magento-style admin console for OneStopShop.",
        analysis_script=_script("analyze_shopping_admin_failures_v2.py"),
        task_families=[
            "Catalog_Product_Admin",
            "Order_Management",
            "Customer_Management",
            "Promotion_Rules",
            "Content_CMS",
            "Store_Configuration",
            "Reports_Analytics",
        ],
        verified_selectors="""
Navigation and search:
- Admin menu items: `a:has-text('Catalog')`, `a:has-text('Sales')`, `a:has-text('Customers')`, `a:has-text('Marketing')`
- Global/admin search: `input[type='search'], input[placeholder*='Search']`
- Grid filter inputs: `.admin__control-text, input[name*='filter'], input[placeholder*='Search']`

Admin grids:
- Grid rows: `tbody tr, .data-row`
- Checkboxes: `input[type='checkbox']`
- Action dropdowns: `select[name='action'], button:has-text('Actions')`
- Apply buttons: `button:has-text('Submit'), button:has-text('Apply')`
- Pagination: `button:has-text('Next Page'), a.action-next`

Forms:
- Save buttons: `button:has-text('Save'), button[title='Save']`
- Text inputs: `input.admin__control-text, input[type='text']`
- Textareas: `textarea.admin__control-text, textarea`
- Selects: `select.admin__control-select, select`
- Toggle/switches: `.admin__actions-switch, input[type='checkbox']`

Products/orders/customers:
- Product menu: `a:has-text('Products')`
- Orders menu: `a:has-text('Orders')`
- Customers menu: `a:has-text('All Customers')`
- Status fields: `select[name*='status'], .admin__field-control`
""",
        url_patterns="""
- Dashboard/admin root: `/admin`
- Products grid: admin URLs containing `/catalog/product`
- Orders grid: admin URLs containing `/sales/order`
- Customers grid: admin URLs containing `/customer`
- CMS pages/blocks: admin URLs containing `/cms`
- Marketing/promotions: admin URLs containing `/promo` or `/rule`
""",
        success_generation_guidance="""
Shopping Admin-specific skill patterns:
- Generate skills around Magento admin grid workflows: locate row, apply grid
  filters, open record, edit fields, save, reload, and verify.
- Prefer parameters like `sku`, `order_number`, `customer_email`,
  `coupon_code`, `cms_title`, `field_name`, `field_value`, and `status`.
- Use the following Shopping Admin task patterns as guidance, not as rigid
  routing rules:
  * Back-office reports and analytics queries: best-selling product/category/
    brand rankings, monthly successful order counts, invoice and order payment
    aggregations, top search terms, product view reports, coupons reports, tax
    reports, shipping reports, and best sellers reports. These are normally
    read-only workflows that gather data and compute counts, totals, or ranks.
  * Customer lookup and segmentation: customers with the most orders, phone
    number reverse lookup, customers tied to fraud suspect orders, and customers
    with the most canceled orders. Locate the customer first, then extract
    stable attributes such as name, email, phone, order counts, canceled order
    totals, or recent canceled-order SKUs.
  * Product, inventory, and price management: inventory warnings, out-of-stock
    products, disable products with quality issues, update stock by style/color/
    size, create simple products, adjust prices by amount or percent, mark
    products on sale, and add color or size options.
  * Review and reputation workflows: count reviews by status/date/keyword,
    summarize why customers like or dislike a product, identify dissatisfied
    customers, approve positive reviews, delete negative reviews, delete low-star
    pending reviews, or remove reviews from suspicious accounts.
  * Order fulfillment and service workflows: filter orders by status including
    fraud suspect orders, cancel exact orders, update shipping addresses, add
    tracking numbers with carriers such as FedEx/DHL/UPS/USPS, and send
    customer-facing messages on recent pending orders.
  * Content, display, and marketing configuration: preview themes, edit CMS page titles,
    update product descriptions using review excerpts, and create marketing price rules
    for site-wide discounts or checkout reductions.
  * Search and behavior analysis: analyze top search terms, brands that appear
    frequently among search terms, and product view report signals.
- For admin grid tasks, apply filters before selecting rows and verify the row
  identity from visible cells; never assume the first row is correct.
- For save workflows, wait for admin loading masks to disappear, click Save,
  wait for success messages or page reload, then reopen/filter the record.
- For bulk actions, select every required row, apply the action, and verify bulk
  actions results across all affected records.
- Cover high-frequency admin tasks: sales reports/top-selling products, review
  search and deletion, customer lookup by phone/email, price/stock/status edits,
  disabling products, cancelling orders, adding tracking numbers, editing order
  addresses, notifying customers on pending orders, CMS page title edits, and
  marketing price rules.
""",
        failure_generation_guidance="""
Shopping Admin-specific corrective rules:
- If an admin grid filter failed, generate a skill that fills the filter field,
  applies it, waits for grid reload, and verifies filtered cell values.
- If a failed workflow is a report, search-behavior, review-analysis, or
  customer-statistics task, prefer a read-and-compute skill that gathers the
  required rows across pages and returns the requested count, total, rank, or
  summary instead of changing admin state.
- If save_not_completed, include post-save verification by reloading the record
  or returning to the grid and checking the target field.
- If wrong_row_selected, explicitly match SKU/order/customer/email/title text
  before opening or selecting the row.
- If bulk actions were incomplete, count selected rows, apply the action once,
  and then verify all rows changed or were removed.
- For reports and analytics failures, set the exact report date range/scope,
  refresh the report, and extract table values without changing store state.
- For customer/order lookup failures, filter by stable identifiers such as order
  number, phone, email, SKU, or product name before reading or mutating records.
""",
        example_skill_patterns="""
Example Shopping Admin skill patterns:
- `async def update_product_attribute(page, sku: str, field_label: str, field_value: str) -> bool`
  Filters the product admin grid by SKU, opens the product, updates the field,
  saves, reloads, and verifies the value.
- `async def change_order_status(page, order_number: str, status: str) -> bool`
  Filters orders, opens the exact order, applies status/comment action, and
  verifies the order grid/detail status.
- `async def apply_bulk_action_to_grid_rows(page, grid_name: str, row_texts: list[str], action_name: str) -> int`
  Filters/selects all matching admin grid rows, applies bulk actions, and
  returns the verified affected row count.
- `async def extract_sales_report_metric(page, report_name: str, date_range: str, metric_name: str) -> str`
  Opens reports, configures the date range, refreshes the table, and extracts
  the requested metric.
""",
        operation_failure_examples=[
            "grid_filter_not_applied",
            "save_not_completed",
            "bulk_action_incomplete",
            "pagination_incomplete",
            "selector_miss",
            "async_not_ready",
            "wrong_row_selected",
            "verification_skipped",
        ],
        reasoning_failure_examples=[
            "wrong_product_or_order",
            "wrong_admin_section",
            "wrong_status_value",
            "ignored_constraints",
            "miscount",
            "wrong_date_range",
            "premature_stop",
        ],
        family_keywords={
            "Catalog_Product_Admin": ["catalog", "product", "sku", "stock", "price"],
            "Order_Management": ["order", "invoice", "shipment", "status", "refund"],
            "Customer_Management": ["customer", "account", "group", "address"],
            "Promotion_Rules": ["coupon", "promotion", "cart rule", "discount"],
            "Content_CMS": ["cms", "page", "block", "content"],
            "Store_Configuration": ["configuration", "store", "setting", "tax", "shipping"],
            "Reports_Analytics": ["report", "analytics", "count", "sales", "statistics"],
        },
        object_keywords={
            "product": ["product", "sku", "catalog"],
            "order": ["order", "invoice", "shipment"],
            "customer": ["customer", "account", "group"],
            "promotion": ["coupon", "rule", "discount"],
            "cms_page": ["cms", "page", "block"],
            "configuration": ["setting", "configuration", "store"],
        },
        default_object="admin_records",
        expected_states={
            "Catalog_Product_Admin": "The product grid or product detail page reflects the requested catalog change.",
            "Order_Management": "The order grid or order detail page shows the requested order state.",
            "Customer_Management": "The customer record displays the requested account or group state.",
            "Promotion_Rules": "The promotion or coupon rule is saved and visible.",
            "Content_CMS": "The CMS page or block content reflects the requested edit.",
            "Store_Configuration": "The configuration value is saved and visible after reload.",
            "Reports_Analytics": "The final answer reports the requested report metric accurately.",
        },
        verification_strategies={
            "Catalog_Product_Admin": ["Filter by SKU/title after saving and verify the row value."],
            "Order_Management": ["Open the order detail page and verify status, totals, and comments."],
            "Customer_Management": ["Filter by email/name and verify the customer record."],
            "Promotion_Rules": ["Reload the rule and verify conditions/actions/coupon values."],
            "Content_CMS": ["Open the CMS entity after saving and verify content."],
            "Store_Configuration": ["Reload configuration scope and verify saved value."],
            "Reports_Analytics": ["Use all relevant report rows before answering."],
        },
    ),
    "map": SiteSkillProfile(
        site="map",
        display_name="Map",
        platform_description="OpenStreetMap/Nominatim map and route-planning interface.",
        analysis_script=_script("analyze_map_failures_v2.py"),
        task_families=[
            "Travel_Time_Distance",
            "Multi_Stop_Itinerary",
            "Nearby_POI_Search",
            "Reachability_Check",
            "POI_Attribute_Query",
            "Knowledge_Location_Resolution",
        ],
        verified_selectors="""
Search and results:
- Search input: `input[name='query'], input[type='search'], input[placeholder*='Search']`
- Search button: `button:has-text('Go'), button[type='submit']`
- Search result rows: `.search_results_entry, .result, li:has(a)`
- Result links: `.search_results_entry a, a[href*='/node/'], a[href*='/way/'], a[href*='/relation/']`

Directions:
- Directions link/button: `a:has-text('Directions'), button:has-text('Directions'), a:has-text('Find directions')`
- From input: `input[placeholder*='From'], input[name='route_from']`
- To input: `input[placeholder*='To'], input[name='route_to']`
- Route button: `button:has-text('Go'), button:has-text('Route')`
- Transport mode buttons: `button:has-text('Car'), button:has-text('Foot'), button:has-text('Bicycle')`

Map details:
- POI detail panel: `.browse-section, .details, aside`
- Attribute rows: `dt, dd, tr, .browse-tag-list li`
- Route summary: `.routing_summary, .route-summary, .directions-panel`
""",
        url_patterns="""
- Search: `/?query={query}` or URLs with `search`
- POI pages: `/node/{id}`, `/way/{id}`, `/relation/{id}`
- Directions/routes: URLs or panels containing route from/to parameters
- Map location hash: URLs with `#map={zoom}/{lat}/{lon}`
""",
        success_generation_guidance="""
Map-specific skill patterns:
- Generate skills around map search, POI detail selection, route planning,
  transport mode selection, multi-stop routes, and POI attribute extraction.
- Prefer parameters like `origin`, `destination`, `place_name`, `poi_type`,
  `transport_mode`, `max_minutes`, `attribute_name`, and `waypoints`.
- Use the following Map task patterns as guidance, not as rigid routing rules:
  * Travel time, distance, and route planning: set origin and destination,
    choose walk/drive/bicycle when needed, compare walking vs driving, read
    time/distance with units, and leave the visible route shown for show route
    or get directions tasks.
  * multi-leg itinerary workflows: combine ordered university/place visits,
    or handle mixed-mode tasks such as walking to one stop and then driving to
    another; compute each leg and aggregate the total time or distance.
  * Nearby and nearest POI search: find the nearest POI, nearest few POIs, local
    stores/restaurants/hotels/pharmacies/gas stations, or brand-specific places
    near a reference location; handle constraints such as within 20 minutes,
    within 5 minutes walking, within 50km, or matching a named chain.
  * Reachability and threshold checks: after calculating route time or distance,
    compare the value against the requested limit and return yes/no or the
    filtered candidates that satisfy the threshold.
  * POI attribute lookup: open the correct detail card or page and read address,
    ZIP code, latitude/longitude coordinates, phone, website, opening hours, or
    operator/owner fields from structured labels when available.
  * Map page location and description page tasks: search for the named location,
    open its description page or POI page, and verify the expected place name
    before returning the page URL or leaving it visible.
  * Geography and knowledge-assisted location resolution: resolve indirect destination descriptions
    such as sports team home arenas, historic event locations, movie/TV filming
    schools, Big Apple, top CS school in Massachusetts, state-border questions,
    or cities implied by landmarks before using the map operation.
- For route tasks, always verify origin, destination, transport mode, displayed
  route summary, and units before returning.
- For nearby POI tasks, search relative to the reference location and open the
  POI detail panel when the task asks to show or pull up a page.
- For attribute tasks, verify the POI detail label/value pair instead of reading
  arbitrary page text.
- Cover common Map WebArena tasks: all airports within a driving radius,
  walking-vs-driving comparisons, nearest hotel/store chains, ZIP/address/
  coordinate extraction, state border questions, route pages that must stay
  visible, and multi-leg trips such as walk first then drive.
""",
        failure_generation_guidance="""
Map-specific corrective rules:
- If transport mode failed, generate a skill that explicitly selects walk/drive/
  bike/transit, waits for recalculation, and verifies the route summary changed.
- If a route or directions task failed, regenerate the route with the exact
  origin/destination, keep the route visible for show route tasks, and extract
  the requested time or distance with units.
- If POI detail was not opened, click the correct result and verify a POI detail
  URL/panel appears before returning.
- If multi-stop was incomplete, add every waypoint in order and verify the total
  route summary includes all stops.
- If a nearby POI or threshold task failed, search relative to the reference
  location, evaluate candidates against the requested distance/time limit, and
  return only matching places.
- If an attribute lookup failed, open the POI detail/description page and read
  the exact address, ZIP code, coordinates, phone, website, hours, or operator
  field instead of using unrelated page text.
- If the failure involved units or wrong metric, extract both time and distance
  labels and return only the metric requested by the objective.
- For radius or "within N minutes/km" tasks, evaluate each candidate against the
  threshold instead of listing every visible search result.
- For state-border or indirect geography tasks, treat location resolution as
  reasoning guidance unless the trajectory shows a concrete map operation failure.
""",
        example_skill_patterns="""
Example Map skill patterns:
- `async def calculate_route_time(page, origin: str, destination: str, transport_mode: str) -> str`
  Opens directions, fills origin/destination, selects transport mode, and
  returns the verified route time with units.
- `async def open_poi_detail(page, place_query: str, expected_name: str) -> str`
  Searches, selects the matching result, verifies POI detail panel/URL, and
  returns the current URL.
- `async def find_nearby_pois_within_drive_time(page, origin: str, poi_type: str, max_minutes: int) -> list[str]`
  Searches nearby POIs, checks drive time for candidates, and returns only those
  satisfying the threshold.
- `async def compare_transport_times(page, origin: str, destination: str) -> dict`
  Calculates walking and driving routes separately, verifies both transport
  modes, and returns both times with units.
""",
        operation_failure_examples=[
            "search_not_executed",
            "route_not_calculated",
            "transport_mode_not_selected",
            "poi_not_selected",
            "directions_not_displayed",
            "async_not_ready",
            "wrong_poi_clicked",
            "multi_stop_not_completed",
        ],
        reasoning_failure_examples=[
            "wrong_destination",
            "knowledge_gap",
            "wrong_transport_mode",
            "wrong_metric",
            "unit_confusion",
            "premature_conclusion",
            "wrong_comparison",
            "incomplete_answer",
        ],
        family_keywords={
            "Travel_Time_Distance": ["route", "distance", "time", "drive", "walk"],
            "Multi_Stop_Itinerary": ["multi", "stops", "itinerary", "waypoints", "total"],
            "Nearby_POI_Search": ["nearby", "nearest", "poi", "restaurant", "hotel"],
            "Reachability_Check": ["within", "reachable", "threshold", "can i"],
            "POI_Attribute_Query": ["address", "phone", "hours", "coordinates", "zip"],
            "Knowledge_Location_Resolution": ["where", "location", "description", "landmark"],
        },
        object_keywords={
            "route": ["route", "directions", "distance", "time"],
            "poi": ["poi", "place", "restaurant", "store", "hotel", "park"],
            "address": ["address", "zip", "coordinates", "phone", "hours"],
            "itinerary": ["stops", "waypoints", "itinerary"],
        },
        default_object="map_results",
        expected_states={
            "Travel_Time_Distance": "The route is calculated and the answer reports the requested time or distance.",
            "Multi_Stop_Itinerary": "All required stops are included and the total route value is answered.",
            "Nearby_POI_Search": "The correct nearby POI is selected or listed with supporting map evidence.",
            "Reachability_Check": "The answer compares the route value against the requested threshold.",
            "POI_Attribute_Query": "The requested POI attribute is visible and extracted accurately.",
            "Knowledge_Location_Resolution": "The indirect location is resolved and opened or used in the route.",
        },
        verification_strategies={
            "Travel_Time_Distance": ["Verify origin, destination, transport mode, and units before answering."],
            "Multi_Stop_Itinerary": ["Verify every required waypoint is included in order."],
            "Nearby_POI_Search": ["Verify the POI is near the reference location and satisfies brand/type constraints."],
            "Reachability_Check": ["Compare the displayed time/distance with the exact threshold."],
            "POI_Attribute_Query": ["Open the POI detail panel and verify the attribute label/value."],
            "Knowledge_Location_Resolution": ["Resolve the real-world clue before using the map search."],
        },
    ),
}


def get_site_profile(site: str) -> SiteSkillProfile:
    normalized = site.strip().lower()
    if normalized not in SITE_PROFILES:
        valid = ", ".join(sorted(SITE_PROFILES))
        raise ValueError(f"Unsupported site '{site}'. Valid sites: {valid}")
    return SITE_PROFILES[normalized]


def list_supported_sites() -> List[str]:
    return sorted(SITE_PROFILES)
