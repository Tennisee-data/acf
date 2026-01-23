"""Node.js/Express Stack Profile - Expert guidance for code generation.

This profile is injected into the implementation prompt when the user
selects Node.js as their tech stack. It provides modern ESM patterns,
Express best practices, and TypeScript integration guidance.
"""

PROFILE_NAME = "node"
PROFILE_VERSION = "1.0"

# Technologies covered by this profile
TECHNOLOGIES = ["node", "nodejs", "express", "typescript", "prisma"]

# Injected into the system prompt for implementation
SYSTEM_GUIDANCE = """
## Node.js/Express Expert Guidelines

You are generating Node.js/Express code. Follow these patterns exactly:

### Project Structure (CRITICAL)

```
src/
├── controllers/         # Route handlers (thin controllers)
├── services/           # Business logic
├── middleware/         # Express middleware
├── routes/             # Route definitions
├── models/             # Database models (Prisma schemas)
├── types/              # TypeScript type definitions
├── utils/              # Utility functions
├── config/             # Configuration
├── app.ts              # Express app setup
└── server.ts           # Server entry point
```

### Modern ESM Imports (Node 20+)

```typescript
// CORRECT: ESM imports with .js extension (REQUIRED for compiled TS)
import { UserService } from './services/UserService.js';
import type { User } from './types/User.js';

// WRONG: CommonJS
const express = require('express');

// CORRECT: ESM
import express from 'express';
```

### Express App Setup

```typescript
// app.ts
import express, { Application, Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { errorHandler } from './middleware/errorHandler.js';
import { userRoutes } from './routes/users.js';

const app: Application = express();

// Security middleware
app.use(helmet());
app.use(cors());

// Body parsing
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/users', userRoutes);

// Error handler (MUST be last)
app.use(errorHandler);

export { app };
```

### Route Organization

```typescript
// routes/users.ts
import { Router } from 'express';
import { UserController } from '../controllers/UserController.js';
import { authenticate } from '../middleware/auth.js';
import { validate } from '../middleware/validate.js';
import { createUserSchema } from '../schemas/user.js';

const router = Router();
const controller = new UserController();

router.get('/', controller.list);
router.get('/:id', controller.get);
router.post('/', validate(createUserSchema), controller.create);
router.put('/:id', authenticate, controller.update);
router.delete('/:id', authenticate, controller.delete);

export { router as userRoutes };
```

### Controller Pattern (Thin Controllers)

```typescript
// controllers/UserController.ts
import { Request, Response, NextFunction } from 'express';
import { UserService } from '../services/UserService.js';

export class UserController {
  private userService = new UserService();

  list = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const users = await this.userService.findAll();
      res.json(users);
    } catch (error) {
      next(error); // Pass to error handler
    }
  };

  create = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const user = await this.userService.create(req.body);
      res.status(201).json(user);
    } catch (error) {
      next(error);
    }
  };
}
```

### Service Layer (Business Logic)

```typescript
// services/UserService.ts
import { prisma } from '../config/database.js';
import type { CreateUserInput, User } from '../types/User.js';
import { AppError } from '../utils/AppError.js';

export class UserService {
  async findAll(): Promise<User[]> {
    return prisma.user.findMany();
  }

  async findById(id: string): Promise<User> {
    const user = await prisma.user.findUnique({ where: { id } });
    if (!user) {
      throw new AppError('User not found', 404);
    }
    return user;
  }

  async create(data: CreateUserInput): Promise<User> {
    // Business logic here
    return prisma.user.create({ data });
  }
}
```

### Error Handling (CRITICAL)

```typescript
// utils/AppError.ts
export class AppError extends Error {
  constructor(
    message: string,
    public statusCode: number = 500,
    public isOperational: boolean = true
  ) {
    super(message);
    Error.captureStackTrace(this, this.constructor);
  }
}

// middleware/errorHandler.ts
import { Request, Response, NextFunction } from 'express';
import { AppError } from '../utils/AppError.js';

export function errorHandler(
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
) {
  if (err instanceof AppError) {
    return res.status(err.statusCode).json({
      error: err.message,
    });
  }

  // Log unexpected errors
  console.error(err);

  res.status(500).json({
    error: 'Internal server error',
  });
}
```

### Validation (Zod)

```typescript
// schemas/user.ts
import { z } from 'zod';

export const createUserSchema = z.object({
  body: z.object({
    email: z.string().email(),
    name: z.string().min(1),
    password: z.string().min(8),
  }),
});

// middleware/validate.ts
import { Request, Response, NextFunction } from 'express';
import { AnyZodObject, ZodError } from 'zod';

export const validate = (schema: AnyZodObject) =>
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      await schema.parseAsync({
        body: req.body,
        query: req.query,
        params: req.params,
      });
      next();
    } catch (error) {
      if (error instanceof ZodError) {
        return res.status(400).json({
          error: 'Validation failed',
          details: error.errors,
        });
      }
      next(error);
    }
  };
```

### Prisma Database (PREFERRED)

```typescript
// config/database.ts
import { PrismaClient } from '@prisma/client';

export const prisma = new PrismaClient();

// Graceful shutdown
process.on('beforeExit', async () => {
  await prisma.$disconnect();
});
```

```prisma
// prisma/schema.prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id        String   @id @default(cuid())
  email     String   @unique
  name      String
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
}
```

### Authentication Middleware

```typescript
// middleware/auth.ts
import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { AppError } from '../utils/AppError.js';

export interface AuthRequest extends Request {
  userId?: string;
}

export async function authenticate(
  req: AuthRequest,
  res: Response,
  next: NextFunction
) {
  const token = req.headers.authorization?.split(' ')[1];

  if (!token) {
    return next(new AppError('Authentication required', 401));
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
    req.userId = decoded.userId;
    next();
  } catch {
    next(new AppError('Invalid token', 401));
  }
}
```

### Common Mistakes to Avoid

1. **Not handling async errors** - Always wrap async handlers or use express-async-errors
2. **Putting business logic in controllers** - Keep controllers thin
3. **Not using TypeScript strict mode** - Enable strict in tsconfig.json
4. **Using CommonJS** - Use ESM with proper .js extensions
5. **Forgetting error middleware position** - Must be LAST middleware
6. **Not validating input** - Always validate with Zod or similar

### Required Dependencies

```json
{
  "dependencies": {
    "express": "^4.18.0",
    "cors": "^2.8.5",
    "helmet": "^7.1.0",
    "zod": "^3.22.0",
    "@prisma/client": "^5.7.0",
    "jsonwebtoken": "^9.0.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/node": "^20.0.0",
    "@types/express": "^4.17.0",
    "@types/cors": "^2.8.0",
    "@types/jsonwebtoken": "^9.0.0",
    "prisma": "^5.7.0",
    "tsx": "^4.0.0"
  }
}
```
"""

DEPENDENCIES = [
    "express@^4.18.0",
    "cors@^2.8.5",
    "helmet@^7.1.0",
    "zod@^3.22.0",
]

OPTIONAL_DEPENDENCIES = {
    "database": ["@prisma/client@^5.7.0", "prisma@^5.7.0"],
    "auth": ["jsonwebtoken@^9.0.0", "bcrypt@^5.1.0"],
    "testing": ["vitest@^1.0.0", "supertest@^6.3.0"],
    "validation": ["express-validator@^7.0.0"],
}

TRIGGER_KEYWORDS = [
    "node",
    "nodejs",
    "node.js",
    "express",
    "expressjs",
    "backend javascript",
    "backend js",
    "prisma",
]


def should_apply(tech_stack: list[str] | None, prompt: str) -> bool:
    """Determine if this profile should be applied."""
    prompt_lower = prompt.lower()

    # Check explicit tech stack selection
    if tech_stack:
        tech_lower = [t.lower() for t in tech_stack]
        if any(kw in tech_lower for kw in ["node", "nodejs", "express"]):
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
