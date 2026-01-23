"""React Stack Profile - Expert guidance for code generation.

This profile is injected into the implementation prompt when the user
selects React as their tech stack. It provides modern React patterns,
hooks best practices, and TypeScript integration guidance.
"""

PROFILE_NAME = "react"
PROFILE_VERSION = "1.0"

# Technologies covered by this profile
TECHNOLOGIES = ["react", "typescript", "vite", "tailwindcss", "nextjs"]

# Injected into the system prompt for implementation
SYSTEM_GUIDANCE = """
## React Expert Guidelines

You are generating React code. Follow these patterns exactly:

### Project Structure (CRITICAL)
```
src/
├── components/           # Reusable UI components
│   ├── ui/              # Generic UI (Button, Input, Modal)
│   └── features/        # Feature-specific components
├── hooks/               # Custom hooks
├── pages/               # Route pages (if using file-based routing)
├── lib/                 # Utilities, API clients
├── types/               # TypeScript type definitions
├── stores/              # State management (Zustand/Redux)
└── App.tsx              # Root component
```

### Modern React Patterns (React 18+)

```typescript
// Function components with TypeScript - ALWAYS use this pattern
interface UserCardProps {
  user: User;
  onSelect?: (userId: string) => void;
}

export function UserCard({ user, onSelect }: UserCardProps) {
  // Use const for components, not arrow functions assigned to const
  return (
    <div onClick={() => onSelect?.(user.id)}>
      {user.name}
    </div>
  );
}
```

### Hooks Best Practices (CRITICAL)

```typescript
// useState - Use for local component state
const [count, setCount] = useState(0);
const [user, setUser] = useState<User | null>(null);

// useEffect - Dependencies array is REQUIRED
useEffect(() => {
  fetchUser(userId);
}, [userId]); // ALWAYS include dependencies

// WRONG: Missing dependencies
useEffect(() => {
  fetchUser(userId); // eslint will warn: userId is missing
}, []); // BAD - stale closure

// useCallback - Memoize callbacks passed to children
const handleSubmit = useCallback((data: FormData) => {
  submitForm(data);
}, [submitForm]);

// useMemo - Expensive computations only
const sortedUsers = useMemo(
  () => users.sort((a, b) => a.name.localeCompare(b.name)),
  [users]
);
```

### State Management (Zustand preferred)

```typescript
// stores/userStore.ts - Zustand pattern
import { create } from 'zustand';

interface UserState {
  user: User | null;
  isLoading: boolean;
  login: (credentials: Credentials) => Promise<void>;
  logout: () => void;
}

export const useUserStore = create<UserState>((set) => ({
  user: null,
  isLoading: false,
  login: async (credentials) => {
    set({ isLoading: true });
    const user = await api.login(credentials);
    set({ user, isLoading: false });
  },
  logout: () => set({ user: null }),
}));

// Usage in component
function Profile() {
  const { user, logout } = useUserStore();
  // ...
}
```

### Data Fetching (React Query / TanStack Query)

```typescript
// PREFERRED: Use React Query for server state
import { useQuery, useMutation } from '@tanstack/react-query';

function UserList() {
  const { data: users, isLoading, error } = useQuery({
    queryKey: ['users'],
    queryFn: () => api.getUsers(),
  });

  if (isLoading) return <Spinner />;
  if (error) return <Error message={error.message} />;

  return <ul>{users?.map(user => <li key={user.id}>{user.name}</li>)}</ul>;
}

// Mutations
const mutation = useMutation({
  mutationFn: (newUser: CreateUserData) => api.createUser(newUser),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['users'] });
  },
});
```

### Forms (React Hook Form)

```typescript
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const schema = z.object({
  email: z.string().email('Invalid email'),
  password: z.string().min(8, 'Password must be 8+ characters'),
});

type FormData = z.infer<typeof schema>;

function LoginForm() {
  const { register, handleSubmit, formState: { errors } } = useForm<FormData>({
    resolver: zodResolver(schema),
  });

  const onSubmit = (data: FormData) => {
    // Handle submit
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register('email')} />
      {errors.email && <span>{errors.email.message}</span>}
      <button type="submit">Login</button>
    </form>
  );
}
```

### TypeScript Integration

```typescript
// Always define types for props
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger';
  isLoading?: boolean;
}

// Use generics for reusable components
interface ListProps<T> {
  items: T[];
  renderItem: (item: T) => React.ReactNode;
  keyExtractor: (item: T) => string;
}

function List<T>({ items, renderItem, keyExtractor }: ListProps<T>) {
  return <ul>{items.map(item => <li key={keyExtractor(item)}>{renderItem(item)}</li>)}</ul>;
}
```

### Common Mistakes to Avoid

1. **Missing keys in lists** - Always use unique, stable keys
   ```typescript
   // WRONG: index as key
   items.map((item, index) => <Item key={index} />)
   // CORRECT: unique ID
   items.map(item => <Item key={item.id} />)
   ```

2. **Direct state mutation**
   ```typescript
   // WRONG: mutating state directly
   user.name = 'New Name';
   setUser(user);
   // CORRECT: create new object
   setUser({ ...user, name: 'New Name' });
   ```

3. **Missing useEffect cleanup**
   ```typescript
   useEffect(() => {
     const subscription = api.subscribe();
     return () => subscription.unsubscribe(); // CLEANUP
   }, []);
   ```

4. **Prop drilling** - Use Context or Zustand for deep props

5. **useEffect for derived state** - Use useMemo instead
   ```typescript
   // WRONG: useEffect to compute derived state
   const [fullName, setFullName] = useState('');
   useEffect(() => {
     setFullName(`${firstName} ${lastName}`);
   }, [firstName, lastName]);

   // CORRECT: useMemo
   const fullName = useMemo(() => `${firstName} ${lastName}`, [firstName, lastName]);
   ```

### Tailwind CSS Integration

```typescript
// Use cn() utility for conditional classes
import { cn } from '@/lib/utils';

function Button({ className, variant, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        'px-4 py-2 rounded-md font-medium',
        variant === 'primary' && 'bg-blue-500 text-white',
        variant === 'secondary' && 'bg-gray-200 text-gray-800',
        className
      )}
      {...props}
    />
  );
}
```

### Required Dependencies

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.4.0",
    "react-hook-form": "^7.48.0",
    "@hookform/resolvers": "^3.3.0",
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "vite": "^5.0.0",
    "@vitejs/plugin-react": "^4.0.0",
    "tailwindcss": "^3.3.0"
  }
}
```
"""

DEPENDENCIES = [
    "react@^18.2.0",
    "react-dom@^18.2.0",
    "@tanstack/react-query@^5.0.0",
    "zustand@^4.4.0",
    "react-hook-form@^7.48.0",
    "@hookform/resolvers@^3.3.0",
    "zod@^3.22.0",
]

OPTIONAL_DEPENDENCIES = {
    "routing": ["react-router-dom@^6.20.0"],
    "animation": ["framer-motion@^10.16.0"],
    "icons": ["lucide-react@^0.294.0"],
    "ui": ["@radix-ui/react-slot@^1.0.0", "@radix-ui/react-dialog@^1.0.0"],
    "testing": ["vitest@^1.0.0", "@testing-library/react@^14.1.0"],
}

TRIGGER_KEYWORDS = [
    "react",
    "reactjs",
    "react.js",
    "frontend",
    "vite",
    "nextjs",
    "next.js",
    "jsx",
    "tsx",
    "tailwind",
]


def should_apply(tech_stack: list[str] | None, prompt: str) -> bool:
    """Determine if this profile should be applied."""
    prompt_lower = prompt.lower()

    # Check explicit tech stack selection
    if tech_stack:
        tech_lower = [t.lower() for t in tech_stack]
        if any(kw in tech_lower for kw in ["react", "reactjs", "nextjs", "vite"]):
            return True

    # Check prompt keywords
    return any(kw in prompt_lower for kw in TRIGGER_KEYWORDS)


def get_guidance() -> str:
    """Get the full guidance text to inject into prompts."""
    return SYSTEM_GUIDANCE


def get_dependencies(features: list[str] | None = None) -> list[str]:
    """Get recommended dependencies."""
    deps = DEPENDENCIES.copy()

    if features:
        for feature in features:
            if feature in OPTIONAL_DEPENDENCIES:
                deps.extend(OPTIONAL_DEPENDENCIES[feature])

    return deps
